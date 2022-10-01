#plotting EIM against Mojuco 
# The basic mujoco wrapper.
from dm_control import mujoco
# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib
###
import torch
import torch.utils.data as data
import scipy.interpolate as inter
import re
# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
import dm_control
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation

# General
import copy
import os
import itertools
import numpy as np
import glob
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from ExpectedInformationMaximization import Colors, ModelRecModWithModelVis
import PIL.Image
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from IPython.display import HTML


def substringFinder(words):
    words.sort(key=lambda x:len(x))
    search = words.pop(0)
    s_len = len(search)
    for ln in range(s_len, 0, -1):
        for start in range(0, s_len-ln+1):
            cand = search[start:start+ln]
            for word in words:
                if cand not in word:
                    break
            else:
                return cand
    return False





# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


##############################  Start  ###################################
def display_video(frames, framerate=30, return_anim=False, filename=None):
    #https://github.com/raymond-van/planet/blob/ebe8632967ac9f638b78d6e4fc822e0740b96c86/utils.py
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    if is_notebook():
        # print("displaying in notebook")
        if return_anim:
            return anim
        else: 
            return HTML(anim.to_jshtml())
    else:
        if filename is not None:
           writervideo = animation.FFMpegWriter(fps=60) 
           anim.save(filename+".mp4", writer=writervideo)
        else:
           print("displaying in script")
           return anim

# Check if running from notebook or python script
def is_notebook():
    try:
        if os.environ.get('COLAB_NOTEBOOK_TEST', True):
            return True   # notebook
        else:
            return False # script
    except NameError:
        return False

def display_img(img):
    if type(img) == torch.Tensor:
        fig = plt.figure()
        plt.imshow(img.permute(1,2,0))
    else:
        return PIL.Image.fromarray(img)
    
# Load the environment

#cheetah, hopper, walker, swimmer, humaoid
def xyz2pixels(xyz, camera_matrix):
    """ Project 3D locations to pixel locations using the camera matrix """ 
    #https://github.com/rinuboney/FPAC/blob/b918755d9137fcfb77f2a855db1d3c661444bdf0/utils.py
    xyzs = np.ones((xyz.shape[0], xyz.shape[1]+1))
    xyzs[:, :xyz.shape[1]] = xyz
    xs, ys, s = camera_matrix.dot(xyzs.T)
    x, y = xs/s, ys/s
    return x, y

def get_positions(physics,duration):
    framerate = 60
    timevals = []
    velocity = []
    video_frames = []
    # Simulate and save data
    index_pos= physics.named.data.geom_xpos.axes.row.names
    col_names=['x', 'y', 'z']
    positions=[]
    frames = []
    physics.reset()
    while physics.data.time < duration:
      physics.step()
      timevals.append(physics.data.time)
      velocity.append(physics.data.qvel.copy())
      positions.append(physics.named.data.qpos.copy())
      data_pos=np.zeros((len(index_pos),len(col_names)))
      for index, row_name in enumerate(index_pos):
          for column, column_name in enumerate(col_names):
              data_pos[index,column]=physics.named.data.geom_xpos[row_name, column_name]
      frames.append(data_pos)
      if len(video_frames) < (physics.data.time) * framerate:
         pixels = physics.render(camera_id=1)
         video_frames.append(pixels)

    A = np.stack(frames, axis=2)
    """
    dpi = 100
    width = 480*2
    height = 640
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize)
    _, ax = plt.subplots(2, 3, figsize=figsize, dpi=dpi, sharex=True)
    ax[0,0].plot(timevals, angular_velocity)
    ax[0,0].set_title('velocity')
    ax[0,0].set_ylabel('meter / second')
    for i in range(len(index_pos)):
        ax[1,1].plot(timevals, A[i,0,:])
    ax[1,1].set_xlabel('time (seconds)')
    ax[1,1].set_ylabel('meters')
    _ = ax[1,1].set_title('X')
    for i in range(len(index_pos)):
        ax[0,1].plot(timevals, A[i,1,:])
    ax[0,1].set_xlabel('time (seconds)')
    ax[0,1].set_ylabel('meters')
    _ = ax[0,1].set_title('Y')
    for i in range(len(index_pos)):
        ax[1,0].plot(timevals, A[i,2,:])
    ax[1,0].set_xlabel('time (seconds)')
    ax[1,0].set_ylabel('meters')
    _ = ax[1,0].set_title('height')
    ax[1,2].plot(timevals, stem_height)
    ax[1,2].set_xlabel('time (seconds)')
    ax[1,2].set_ylabel('meters')
    _ = ax[1,2].set_title('position')
    fig.patch.set_visible(False)
    ax[0,2].axis('off')
    plt.tight_layout() 
    plt.show()   
    """
    return positions, velocity, A , display_video(video_frames, framerate)

class MujocoDataset(data.Dataset):

    def __init__(self, observation, next_observation):
        super(MujocoDataset, self).__init__()
        self.dataset1 = observation
        self.dataset2 = next_observation

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        obs1=self.dataset1[index]
        obs2=self.dataset2[index]
        return obs1, obs2

def shufflerow(tensor1, tensor2, axis, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    row_perm = torch.rand(tensor1.shape[:axis+1]).argsort(axis)  # get permutation indices
    for _ in range(tensor1.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor1.shape[axis+1:]))  # reformat this for the gather operation
    return tensor1.gather(axis, row_perm.to(device)),tensor2.gather(axis, row_perm.to(device))

def split_data(data,valid_portion,dim):
    indices=torch.randperm(data[0].shape[dim])
    valid_size=int(len(indices)*valid_portion)
    train_size=len(indices)-valid_size
    train_data=(data[0][:train_size,:],data[1][:train_size,:])
    valid_data=(data[0][train_size:,:],data[1][train_size:,:])
    return train_data, valid_data 
    
class MujocoData:

    def __init__(self, observation, next_observation, num_obstacles=2, train_valid_split_portion=7/8, samples_per_context=None, seed=0):
        self.data =(observation, next_observation)
        self._num_obstacles = num_obstacles
        self._rng = np.random.RandomState(seed)
        self._num_obstacles = num_obstacles
        self._context_dim = 2 * num_obstacles
        self._sample_dim = 2 * num_obstacles #+ 2
        if samples_per_context is not None:
           self._samples_per_context = samples_per_context
        else:
          self._samples_per_context = observation.shape[0]//2
        #mujoco_dataset=MujocoDataset(observation, next_observation)
        #self.train_samples =data.DataLoader(mujoco_dataset, batch_size=1, shuffle=True)
        training_samples, validation_samples=split_data((observation,next_observation),1-train_valid_split_portion,0)
        self.train_samples = training_samples
        #self.test_samples = data.DataLoader(mujoco_dataset, batch_size=1, shuffle=True)
        self.test_samples = shufflerow(training_samples[0], training_samples[1], 0)

        self.val_samples = validation_samples
        
    def get_spline(self, x):
        x_ext = np.zeros(2 * self._num_obstacles + 4, dtype=x.dtype)
        x_ext[0] = 0.0
        x_ext[1] = 0.5
        x_ext[2:-2] = (x + 1) / 2
        x_ext[-2] = 1.0
        x_ext[-1] = 0.5 #'(x[-1] + 1) / 2
        k = "quadratic" if self._num_obstacles == 1 else "cubic"
        return inter.interp1d(x_ext[::2], x_ext[1::2], kind=k)


class MujocoModelRecMod(ModelRecModWithModelVis):
      def __init__(self, mujoco_data, env_name, train_samples, test_samples, true_log_density=None, eval_fn=None,
                 test_log_iters=50, save_log_iters=50,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__( train_samples, test_samples, true_log_density, eval_fn, test_log_iters, save_log_iters)
        self._data = mujoco_data.data
        self._env_name= env_name
        self._test_samples = test_samples
        self.device = device
        self.environment=None
        """Returns a tuple containing the model XML string and a dict of assets."""
        root_dir = os.path.dirname(dm_control.__file__)+'/suite/'
        #root_dir = '/usr/local/lib/python3.7/dist-packages/dm_control/suite/'
        for files in glob.glob(os.path.join(root_dir, '*.xml')):
            base=os.path.basename(files)
            filename=os.path.splitext(base)[0]
            if filename in env_name.split('-')[0].lower():
               self.environment=files
               #print(f" deepmind control env: {self.environment}")      
      def get_model_and_assets(self):
          if self.environment is not None:
              xml = resources.GetResource(self.environment)
              return xml, common.ASSETS
          else: 
             raise ValueError('There is no environment here.')
         
      def gt_keypoints(self, h, w):
          """ Extract 2D pixel locations of objects in the environment """
          camera_matrix = mujoco.Camera(self.physics, height=h, width=w, camera_id=1).matrix
          xyz = self.physics.named.data.geom_xpos.copy()
          """
          head_pos = self.physics.named.data.geom_xpos['head']
          head_mat = self.physics.named.data.geom_xmat['head'].reshape(3, 3)
          head_size = self.physics.named.model.geom_size['head']
          offsets = np.array([-1, 1]) * head_size[:, None]
          xyz_local = np.stack(itertools.product(*offsets)).T
          xyz_global = head_pos[:, None] + head_mat @ xyz_local   
          # # Camera matrices multiply homogenous [x, y, z, 1] vectors.
          corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
          corners_homogeneous[:3, :] = xyz_global
          uh, vh, zh = camera_matrix @ corners_homogeneous
          # x and y are in the pixel coordinate system of the head.
          xh = uh / zh
          yh = vh / zh
          max_width = physics.model.vis.global_.offwidth
          max_height = physics.model.vis.global_.offheight
          mujoco.MovableCamera(physics, height=max_height, width=max_width)
          """
          return xyz2pixels(xyz, camera_matrix)

      def _plot_model(self, model, title):
            x_plt = np.arange(0, 1, 1e-2)
            color = Colors()
            self.physics = mujoco.Physics.from_xml_string(*self.get_model_and_assets())
            objects  =[]
            pixels   =[]
            # Visualize the joint axis.
            scene_option = mujoco.wrapper.core.MjvOption()
            scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True
            observation, next_observation=self._data
            with self.physics.reset_context():
               """Returns a copy of the generalized positions (system configuration)."""
               #position=physics.data.qpos[:]
               pos_size = self.physics.data.qpos[:].shape[0]
               print(f"size of data {self.physics.data.qpos[:].shape}, {observation.shape}")
               """Returns a copy of the generalized velocities."""
               #velocity=physics.data.qvel[:]
               #observation = np.concatenate([position, velocity]).ravel()
               position=self.physics.named.data.geom_xpos.copy()
               print(f"size of position {position.shape}")
               self.physics.named.data.geom_xpos[:] = observation[:pos_size, ...]
               pixels.append(self.physics.render(scene_option=scene_option, camera_id=1, segmentation=True))
               height, width, _ = pixels[0].shape
               """Observations consist of an OrderedDict containing one or more NumPy arrays"""
               objects.append( self.gt_keypoints(height, width))
               self.physics.named.data.geom_xpos[:] = next_observation[:pos_size, ...]
               pixels.append(self.physics.render(scene_option=scene_option, camera_id=1, segmentation=True))
               objects.append( self.gt_keypoints(height, width))
            fig, ax = plt.subplots(1, 1)
            PIL.Image.fromarray(pixels[0][:,:,0])
            if isinstance(self._test_samples,tuple):
               contexts=tuple(item.to(device=self.device, dtype=torch.float32) for item in self._test_samples)
            else:
               contexts = self._test_samples.to(device=self.device,dtype=torch.float32)
            for i in range(len(contexts)):
                context= contexts[i]
                lines=[]
                for k, c in enumerate(model.components):
                    #needs to be debugged here 
                    m = (c.mean(context)[0] + 1) / 2
                    print(m.shape)
                    cov = c.covar(context)[0]
                    mx, my = m[::2].cpu().detach().numpy(), m[1::2].cpu().detach().numpy()
                    plt.scatter(200 * mx, 100 * my, c=color(k))
                    for j in range(mx.shape[0]):
                       mean = np.array([mx[j], my[j]])
                       cov_j = cov[2 * j: 2 * (j + 1), 2 * j: 2 * (j + 1)].cpu().detach().numpy()
                       plt_cx, plt_cy = self._draw_2d_covariance(mean, cov_j, 1, return_raw=True)
                       plt.plot(200 * plt_cx, 100 * plt_cy, c=color(k), linestyle="dotted", linewidth=2)
                    for j in range(2):
                        s = np.array(c.sample(contexts[i].cpu().detach().numpy()))
                        spline = self._data.get_spline(s[0])
                        l, = plt.plot(200 * x_plt, 100 * spline(x_plt), c=color(k), linewidth=1)
     
                lines.append(l)
                weights = model.gating_distribution.probabilities(context)[0]
                strs = ["{:.3f}".format(weights[j]) for j in range(model.num_components)]
                plt.legend(lines, strs, loc=1)
                plt.gca().set_axis_off()
                plt.gca().set_xlim(0, 200)
                plt.gca().set_ylim(0, 100)
            plt.savefig("EIM_out_imgs/" + re.sub(r"\s+", '-', title) + ".png")
