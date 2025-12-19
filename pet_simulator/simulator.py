import os

from skimage.transform import radon
from skimage.filters import gaussian

from scipy.signal import convolve2d as conv2

import numpy as np
import cv2
from tools.image.castor import read_castor_binary_file, write_binary_file



class SinogramSimulator:

    """
    A toy pet simulation utility.
    Mathematical Randon transform is used to simulate the projection.
    Poisson noise is then added to the projection data.
    The simulation also provides materials to perform the inversion.
    """

    def __init__(self, n_angles=344, nb_radius_px=252, voxel_size_mm=(2.0, 2.0), random_deficiencies=5.0,seed=None):
        """
        param angles: angles to use for the radon transform (in degrees).
        param seed: random seed for noise generation.
        """
        if seed is not None:
            self.set_seed(seed)
        if isinstance(voxel_size_mm, (int, float)):
            self.voxel_size_mm2 = voxel_size_mm ** 2
        elif isinstance(voxel_size_mm, (list, tuple)) and len(voxel_size_mm) == 2:
            self.voxel_size_mm2 = voxel_size_mm[0] * voxel_size_mm[1]
        else:
            print("No voxel size provided, assuming 1mm x 1mm.")
            self.voxel_size_mm2 = 1.0
        self.voxel_size_mm = voxel_size_mm
        self.random_deficiencies = random_deficiencies

        self.n_angles = n_angles
        self.angles = angles=np.arange(0,180, 180 / self.n_angles)
        self.nb_radius_px = nb_radius_px

        self.scatter_gaussian_sigma = 2.0  # in pixels
        self.variance_random = 1.0  # variance for randoms noise model

    def set_seed(self, seed):
        np.random.seed(seed)

    def simulate(self, img_path, img_att_path=None,
        nb_count=3e6,
        scatter_component=0.35,
        random_component=0.40,
        gaussian_PSF=4 # in mm
        ):

        # Read images
        img_object = read_castor_binary_file(img_path)
        img_object = np.array(img_object, dtype=np.float32)
        img_object = img_object.squeeze()
        if img_att_path is not None:
            img_att = read_castor_binary_file(img_att_path)
            img_att = np.array(img_att, dtype=np.float32)
            img_att = img_att.squeeze()
        else:
            img_att = None

        assert len(img_object.shape) == 2, "Only 2D images are supported."
        assert img_att is None or img_att.shape == img_object.shape, "Attenuation image must have the same shape as object image."

        # PSF in image domain
        if gaussian_PSF > 0.0:
            sigma_mm = gaussian_PSF / (2 * np.sqrt(2 * np.log(2)))
            img_object = gaussian(img_object, sigma=(sigma_mm / self.voxel_size_mm[0], sigma_mm / self.voxel_size_mm[1]))

        # compute true counts sinogram
        true_counts = radon(img_object, theta=self.angles, circle=False)

        # apply attenuation if provided
        if img_att is not None:
            attenuation_scale_factor = 0.01 # cm^-1 to mm^-1
            att_sino = radon(img_att * attenuation_scale_factor, theta=self.angles, circle=False)
            # apply attenuation
            true_counts = true_counts * np.exp(-att_sino * self.voxel_size_mm2)

        # transpose true counts to match castor behavior
        true_counts = np.transpose(true_counts)

        # resize to match target binning
        voxel_size_ratio = true_counts.shape[1] / self.nb_radius_px
        true_counts = cv2.resize(true_counts, (self.nb_radius_px, self.n_angles), interpolation=cv2.INTER_LINEAR)

        # PSF in sinogram domain
        if gaussian_PSF > 0.0:
            pass
            # 1D convoltion on each row
            for j in range(true_counts.shape[0]):
                angle = self.angles[j]
                bin_width_mm = np.abs(np.cos(angle * np.pi / 180)) * self.voxel_size_mm[0] * voxel_size_ratio + np.abs(np.sin(angle * np.pi / 180)) * self.voxel_size_mm[1] * voxel_size_ratio
                sigma_bins = sigma_mm / bin_width_mm
                true_counts[j, :] = gaussian(true_counts[j, :], sigma=sigma_bins)

        # scatter model
        if scatter_component > 0:
            sino_scatter = gaussian(true_counts, sigma=(self.scatter_gaussian_sigma, self.scatter_gaussian_sigma), mode='reflect')
            sino_scatter = (sino_scatter / sino_scatter.sum()) * (scatter_component * true_counts.sum())
        else:
            sino_scatter = np.zeros_like(true_counts)

        # random model
        if random_component > 0:
            sino_random = np.random.randn(*true_counts.shape).astype(np.float32) * self.variance_random + self.random_deficiencies
            sino_random = np.clip(sino_random, a_min=0, a_max=None)
            sino_random = (sino_random / sino_random.sum()) * (random_component * true_counts.sum())
        else:
            sino_random = np.zeros_like(true_counts)

        # compute noise-free prompt sinogram
        noise_free_prompt = true_counts + sino_scatter + sino_random

        # scale to desired total true counts
        scale = (nb_counts * (obj_size / average_size)) / noise_free_prompt.sum()
        noise_free_prompt = np.round(noise_free_prompt * scale)
        true_counts = true_counts * scale
        sino_scatter = sino_scatter * scale
        sino_random = sino_random * scale

        # add Poisson noise
        prompt = np.random.poisson(noise_free_prompt)

        return true_counts, sino_scatter, sino_random, noise_free_prompt, prompt

    def run(self, img_path, img_att_path, dest_path, simulate_args={}):

        #
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        self.dout = dest_path

        # Simulation
        _, sino_scatter, sino_random, noise_free_prompt, prompt = self.simulate(img_path, img_att_path, **simulate_args)

        # Save data with Castor conventions
        # cast prompt to int16
        prompt = prompt.astype(np.int16)

        for filename, data in zip(['simu_sc.s', 'simu_rd.s', 'simu_nfpt.s', 'simu_pt.s'], [sino_scatter, sino_random, noise_free_prompt, prompt]):
            write_binary_file(os.path.join(self.dout, 'simu', filename), data, binary_extension='')

if __name__ == "__main__":

    from phantom_simulation.object_simulator import Phantom2DPetGenerator
    import numpy as np
    import os

    generator = Phantom2DPetGenerator(shape=(256,256), voxel_size=(2,2,2))
    generator.set_seed(42)
    obj_path, att_path = generator.run(os.path.join(os.getenv('WORKSPACE'), 'data/data1/object'))

    simulator = SinogramSimulator()
    simulator.set_seed(42)

    # true_counts, scatter, random, noise_free_prompt, prompt = simulator.simulate(img_path=obj_path, img_att_path=att_path, nb_count=3e6)
    simu_dest_path=os.path.join(os.getenv('WORKSPACE'), 'data/data1/simulation')
    simulator.run(img_path=obj_path, img_att_path=att_path, dest_path=simu_dest_path, simulate_args={'nb_count': 3e6})

    scatter = read_castor_binary_file(os.path.join(simu_dest_path, 'simu_sc.hdr')).squeeze()
    random = read_castor_binary_file(os.path.join(simu_dest_path, 'simu_rd.hdr')).squeeze()
    nf_prompt = read_castor_binary_file(os.path.join(simu_dest_path, 'simu_nfpt.hdr')).squeeze()
    prompt = read_castor_binary_file(os.path.join(simu_dest_path, 'simu_pt.hdr')).squeeze()


    import matplotlib.pyplot as plt


    fig, ax = plt.subplots(2, 2)
    ax[0,0].set_title('Scater')
    im0 = ax[0,0].imshow(scatter, cmap='gray')
    fig.colorbar(im0, ax=ax[0, 0])
    ax[0,1].set_title('Random')
    im1 = ax[0,1].imshow(random, cmap='gray')
    fig.colorbar(im1, ax=ax[0, 1])
    ax[1,0].set_title('Noise-free Prompt')
    im2 = ax[1,0].imshow(nf_prompt, cmap='gray')
    fig.colorbar(im2, ax=ax[1, 0])
    ax[1,1].set_title('Prompt Sinogram ')
    im3 = ax[1,1].imshow(prompt, cmap='gray')
    fig.colorbar(im3, ax=ax[1, 1])
    plt.show()
