# Code adapted from
#
# Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023).
# Improving and generalizing flow-based generative models with minibatch optimal transport.
# arXiv preprint arXiv:2302.00482.
# (https://github.com/atong01/conditional-flow-matching)
#
# Chemseddine, J., Hagemann, P., Wald, C., & Steidl, G. (2024).
# Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching.
# arXiv preprint arXiv:2403.18705.
# (https://github.com/JChemseddine/Conditional_Wasserstein_Distances/blob/main/utils/utils_FID.py)

import os
import numpy as np
import ot
from tqdm import tqdm
from matplotlib import pyplot as plt  # not really used but left in case
import torch
from torchdiffeq import odeint_adjoint as odeint

import pnpflow.fid_score as fs  # kept in case you want to use it later
from pnpflow.dataloaders import DataLoaders
import pnpflow.utils as utils
from pnpflow.dataloaders import CelebADataset, AFHQDataset
from torchvision import transforms
from torchvision.utils import save_image

# NOTE: img_dir_afhq had a missing slash before 'data'; left as-is if repo expects it,
# but you may want './data/afhq_cat/test/cat/' instead.
img_dir_celeba = './data/celeba/img_align_celeba/img_align_celeba'
partition_csv_celeba = './data/celeba/list_eval_partition.csv'
img_dir_afhq = '.data/afhq_cat/test/cat/'


class FLOW_MATCHING(object):

    def __init__(self, model, device, args):
        self.d = args.dim_image
        self.num_channels = args.num_channels
        self.device = device
        self.args = args
        self.lr = args.lr
        self.model = model.to(device)

    def train_FM_model(self, train_loader, opt, num_epoch):

        # --- REMOVED: fld-based FID + Inception feature extractor ----------------
        # Previously:
        # ft_extractor = InceptionFeatureExtractor(save_path="features")
        # if self.args.dataset == "celeba":
        #     test_feat = ft_extractor.get_features(...)
        # elif self.args.dataset == "afhq_cat":
        #     test_feat = AFHQDataset(...)
        # else:
        #     raise ValueError(...)
        # -------------------------------------------------------------------------

        tq = tqdm(range(num_epoch), desc='loss')
        for ep in tq:
            for iteration, (x, labels) in enumerate(train_loader):
                if x.size(0) == 0:
                    continue
                if iteration > 20:
                    break
                print(f'Epoch: {ep}, iter: {iteration}')
                x = x.to(self.device)
                z = torch.randn(
                    x.shape[0],
                    self.num_channels,
                    self.d,
                    self.d,
                    device=self.device,
                    requires_grad=True
                )

                t1 = torch.rand(x.shape[0], 1, 1, 1, device=self.device)

                # compute coupling
                x0 = z.clone()
                x1 = x.clone()
                a, b = np.ones(len(x0)) / len(x0), np.ones(len(x0)) / len(x0)

                M = ot.dist(x0.view(len(x0), -1).cpu().data.numpy(),
                            x1.view(len(x1), -1).cpu().data.numpy())
                plan = ot.emd(a, b, M)
                p = plan.flatten()
                p = p / p.sum()
                choices = np.random.choice(
                    plan.shape[0] * plan.shape[1], p=p, size=len(x0), replace=True)
                i, j = np.divmod(choices, plan.shape[1])
                x0 = x0[i]
                x1 = x1[j]
                xt = t1 * x1 + (1 - t1) * x0
                loss = torch.sum(
                    (self.model(xt, t1.squeeze()) - (x1 - x0))**2) / x.shape[0]
                opt.zero_grad()
                loss.backward()
                opt.step()

                # save loss in txt file
                with open(self.save_path + 'loss_training.txt', 'a') as file:
                    file.write(
                        f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n'
                    )

            # save samples & plots
            self.sample_plot(x, ep)

            if ep % 5 == 0:
                # save model
                torch.save(self.model.state_dict(),
                           self.model_path + f'model_{ep}.pt')

                # --- REMOVED: FID computation using fld.FID ----------------------
                # print("Computing FID 5K")
                # num_gen = 5_000
                # fid_value = self.compute_fid(
                #     num_gen, test_feat, ft_extractor,
                #     batch_size=124, integration_method="euler",
                #     integration_steps=10
                # )
                # with open(self.save_path + f'FID_{(num_gen // 1000)}k.txt', 'a') as file:
                #     file.write(f'Epoch: {ep}, FID: {fid_value}\n')
                # -----------------------------------------------------------------
                print("Skipping FID computation (fld module not available).")

    def apply_flow_matching(self, NO_samples):
        self.model.eval()
        with torch.no_grad():
            model_class = cnf(self.model)
            latent = torch.randn(
                NO_samples,
                self.num_channels,
                self.d,
                self.d,
                device=self.device,
                requires_grad=False
            )
            z_t = odeint(
                model_class,
                latent,
                torch.tensor([0.0, 1.0]).to(self.device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )
            x = z_t[-1].detach()
        self.model.train()
        return x

    def sample_plot(self, x, ep=None):
        try:
            os.makedirs(self.save_path + 'results_samplings/')
        except BaseException:
            pass

        reco = utils.postprocess(self.apply_flow_matching(16), self.args)
        reco = utils.postprocess(reco, self.args)
        utils.save_samples(
            reco.detach().cpu(),
            x[:16].cpu(),
            self.save_path + 'results_samplings/' +
            f'samplings_ep_{ep}.pdf',
            self.args
        )

        # check the plots by saving training samples
        if ep == 0:
            gt = x[:16]
            gt = utils.postprocess(gt, self.args)
            utils.save_samples(
                gt.detach().cpu(),
                gt.detach().cpu(),
                self.save_path + 'results_samplings/' +
                f'train_samples_ep_{ep}.pdf',
                self.args
            )

    def generate_samples(self, integration_method="dopri5", tol=1e-5,
                         n_samples=1028, batch_size=None, num_channels=3,
                         integration_steps=100, tmax=1):
        """
        Return a tensor of generated images.
        """

        if batch_size is None:
            batch_size = n_samples

        images_list = []
        batches = [batch_size] * (n_samples // batch_size)
        if n_samples % batch_size:
            batches += [n_samples % batch_size]

        with torch.no_grad():
            for k, batch in enumerate(tqdm(batches)):
                time_points = torch.linspace(
                    0, tmax, int(tmax * integration_steps), device=self.device
                )

                x0 = torch.randn(batch, num_channels, self.d,
                                 self.d, device=self.device)
                model_class = cnf(self.model)
                traj = odeint(
                    model_class,
                    x0,
                    time_points,
                    rtol=tol,
                    atol=tol,
                    method=integration_method
                )
                images_list.append(traj[-1, :])

        images = torch.cat(images_list, dim=0)
        return images

    # --- OPTIONAL: stub of compute_fid kept for compatibility ---------------
    # If something elsewhere calls self.compute_fid, this will not crash,
    # but just return NaN and save some images.
    def compute_fid(self, num_images_fid, train_feat, ft_extractor,
                    batch_size=512, integration_method="dopri5",
                    integration_steps=100, epoch='final'):
        print("compute_fid called, but FID computation is disabled (fld missing).")
        gen_images = self.generate_samples(
            integration_method=integration_method,
            tol=1e-4,
            n_samples=num_images_fid,
            batch_size=batch_size,
            integration_steps=integration_steps
        )

        # save the 16 first generated images in a grid
        os.makedirs(f"training_images/{self.args.dataset}", exist_ok=True)
        images = gen_images[:16]
        save_image(
            images,
            f"training_images/{self.args.dataset}/gen_images_epoch{epoch}.png"
        )

        return float("nan")
    # ------------------------------------------------------------------------

    def train(self, data_loaders):

        self.save_path = self.args.root + \
            'results/{}/ot/'.format(self.args.dataset)
        try:
            os.makedirs(self.save_path)
        except BaseException:
            pass

        self.model_path = self.args.root + \
            'model/{}/ot/'.format(self.args.dataset)
        try:
            os.makedirs(self.model_path)
        except BaseException:
            pass

        # load model
        train_loader = data_loaders['train']

        # create txt file for storing all information about model
        with open(self.save_path + 'model_info.txt', 'w') as file:
            file.write(f'PARAMETERS\n')
            file.write(
                f'Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n'
            )
            file.write(f'Number of epochs: {self.args.num_epoch}\n')
            file.write(f'Batch size: {self.args.batch_size_train}\n')
            file.write(f'Learning rate: {self.lr}\n')

        # start training
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_FM_model(train_loader, opt, num_epoch=self.args.num_epoch)

        # save final model
        torch.save(self.model.state_dict(), self.model_path + 'model_final.pt')


class cnf(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        with torch.no_grad():
            # z = self.model(x, t.squeeze())
            z = self.model(x, t.repeat(x.shape[0]))
        return z
