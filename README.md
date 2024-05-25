# python-guidelines

Guidelines for code style with Python in Sombra Studio. For the most part, just
use Google Python style https://google.github.io/styleguide/pyguide.html

## General

### 1. Lines should have a maximum of 80 characters
### 2. When the content inside of parenthesis is longer than 80 characters, use new line with an indentation. And then, finish with the closing parenthesis at the same level.

That also goes for arrays and dictionaries.

#### Do this

```py
label_params = LabelParams(
    label_x,
    label_y,
    self.width,
    self.height,
    value=self.label_value,
    style=params.style
)
```

#### Don't do this

```py
label_params = LabelParams(label_x,
                            label_y,
                            self.width,
                            self.height,
                            value=self.label_value,
                            style=params.style)
```

#### Why

Because the second method is inconsistent, depending on where the parenthesis 
starts you would have


## Naming conventions

### 1. Use all upper case letters for naming constants

```py
COLOR_CHANNELS = 1
MAX_COLOR = 255
V_SAMPLES = 3
H_SAMPLES = 3
```
### 2. Use snake_case for functions, methods and variable names

#### Do this

```py
def find_closest_color(color, palette):
    min_difference = np.inf
    closest_color = palette[0]
    for palette_color in palette:
        difference = np.abs(color - palette_color)
        if difference < min_difference:
            min_difference = difference
            closest_color = palette_color
    return closest_color
```

#### Don't do this

```py
def findClosestColor(color, palette):
    minDifference = np.inf
    closest_color = palette[0]
    for paletteColor in palette:
        difference = np.abs(color - paletteColor)
        if difference < minDifference:
            minDifference = difference
            closest_color = paletteColor
    return closest_color
```

### 3. Use CamelCase for class names

#### Do this

```py
class VertexGroup:
```


## Functions

### 1. Should encapsulate one thing, and be concise.
### 2. You should be able to view the entire definition in the screen without the need of scrolling down or right (around 40 lines and 80 columns max).

#### Do this

```py

def update_particles(dt):
    if paused:
        return
    global emission_x, emission_y, timer
    particle_system.update(dt)
    # Move emitter from input
    if keys[key.LEFT]:
        emission_x -= SPEED_X * dt
    elif keys[key.RIGHT]:
        emission_x += SPEED_X * dt
    if keys[key.UP]:
        emission_y += SPEED_Y * dt
    elif keys[key.DOWN]:
        emission_y -= SPEED_Y * dt
    # Emit particles
    timer += dt
    if timer > EMISSION_RATE:
        timer = 0
        start_color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        end_color = (start_color * 0.25)
        start_m = np.random.uniform(1 / MASS_SCALE, MASS_SCALE)
        end_m = start_m / MASS_SCALE

        particle_settings = ParticleSettings(
            start_color, end_color,
            START_OPACITY, END_OPACITY,
            MIN_LIFESPAN, MAX_LIFESPAN,
            start_m, end_m
        )
        particle_system.emit(
            emission_x, emission_y, EMISSION_COUNT, particle_settings,
            MIN_START_VEL, MAX_START_VEL
        )
```

#### Don't do this

```py
def train(
    self, svm_kernel, codebook, des_option=constants.ORB_FEAT_OPTION,
    is_interactive=True
):
    if des_option == constants.ORB_FEAT_OPTION:
        des_name = constants.ORB_FEAT_NAME
    else:
        des_name = constants.SIFT_FEAT_NAME
    k = len(codebook)
    x_filename = filenames.vlads_train(k, des_name)
    if is_interactive:
        data_option = input(
            "Enter [1] to calculate VLAD vectors for the training set or"
            "[2] to load them.\n"
        )
    else:
        data_option = constants.GENERATE_OPTION
    if data_option == constants.GENERATE_OPTION:
        # Getting the global vectors for all of the training set
        print("Getting global descriptors for the training set.")
        start = time.time()
        x, y = self.get_data_and_labels(
            self.dataset.get_train_set(), codebook, des_option
        )
        utils.save(x_filename, x)
        end = time.time()
        print("VLADs training vectors saved on file {0}".format(x_filename))
        self.log.train_vlad_time(end - start)
    else:
        # Loading the global vectors for all of the training set
        print("Loading global descriptors for the training set.")
        x = utils.load(x_filename)
        y = self.dataset.get_train_y()
        x = np.matrix(x, dtype=np.float32)
    svm = cv.ml.SVM_create()
    svm_filename = filenames.svm(k, des_name, svm_kernel)
    if is_interactive:
        svm_option = input(
            "Enter [1] for generating a SVM or [2] to load one\n"
        )
    else:
        svm_option = constants.GENERATE_OPTION
    if svm_option == constants.GENERATE_OPTION:
        # Calculating the Support Vector Machine for the training set
        print(
            "Calculating the Support Vector Machine for the training set..."
        )
        svm.setKernel(svm_kernel)
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setC(1)
        svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        start = time.time()
        svm.train(x, cv.ml.ROW_SAMPLE, y)
        end = time.time()
        self.log.svm_time(end - start)
        # Storing the SVM in a file
        svm.save(svm_filename)
    else:
        svm.load(svm_filename)
    return svm
```

### 3. Should have maximum 8 parameters (if possible)

#### Do this

```py
class Button(Widget):
    # Here we use a params object that will hold arguments for the init
    def __init__(
        self,
        params: ButtonParams,
        batch: pyglet.graphics.Batch,
        group: pyglet.graphics.Group = None
    ):
        super().__init__(params)
        self.label_value: str = params.label
        self.on_press = params.on_press
        self.color: [int, int, int, int] = params.color
        self.hover_color: [int, int, int, int] = params.hover_color
        self.press_color: [int, int, int, int] = params.press_color
        self.batch: pyglet.graphics.Batch = batch
        self.group: pyglet.graphics.Group = group
        self.background: Rectangle = self.create_background()
        # define label params
        label_x = self.x + self.width / 2.0
        label_y = self.y + self.height / 2.0
        label_params = LabelParams(
            label_x,
            label_y,
            self.width,
            self.height,
            value=self.label_value,
            style=params.style
        )
        self.label: Label = Label(label_params, batch=batch, group=group)
```

### Don't do this
```py
class Button(Widget):
    # Notice the amount of arguments in this init function
    def __init__(
        self, x, y, width, height, label, on_press, color, hover_color, 
        press_color, style,
        batch,
        group = None
    ):
        super().__init__(x, y, width, height, batch, group)
        self.label_value: str = label
        self.on_press = on_press
        self.color: [int, int, int, int] = color
        self.hover_color: [int, int, int, int] = hover_color
        self.press_color: [int, int, int, int] = press_color
        self.batch: pyglet.graphics.Batch = batch
        self.group: pyglet.graphics.Group = group
        self.background: Rectangle = self.create_background()
        # define label params
        label_x = self.x + self.width / 2.0
        label_y = self.y + self.height / 2.0
        label_params = LabelParams(
            label_x,
            label_y,
            self.width,
            self.height,
            value=self.label_value,
            style=style
        )
        self.label: Label = Label(label_params, batch=batch, group=group)
```


## Docstrings

Use the Google format [https://google.github.io/styleguide/pyguide.html#381-docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings)

```py
class Camera:
    def __init__(
        self,
        position=np.array([512.0, 88.0, 512.0]),
        up=np.array([0.0, 1.0, 0.0]),
        theta=pi/2, z_far=1000, fov=90.0, proj_dist=1.0, proj_height=0.35,
        horizon=HORIZON
    ):
        """
        Initialize a camera

        Args:
            position (ndarray): The position in 3D space
            up (ndarray): Unit vector representing the up direction in 3D space
            theta (float): Angle of the camera in the y-axis given in rads
            z_far (float): Limit to where how far the camera can see in the
                terrain in camera coordinates. z_far CANNOT be greater than the
                texture size
            fov (float): Field of view given in degrees
            proj_dist (float): Distance to the projection window
            proj_height (float): Height of the viewing projection window in
                world's units
            horizon (float): Offset height in pixels at which that scrolls the
                projected image up or down
        """
```

## Imports

When importing modules use alphabetical order and import global packages first
and then local modules. Sometimes import order matters, only in those case don't
use alphabetical and add a comment explaining that.

```py
import numpy as np
from PIL import Image

# Local modules
import utils
```

```py
from .mesh import Mesh
from .model import Model
from .wireframe import Wireframe
# IMPORTANT DON'T MOVE THE IMPORT OF OBJ, IT NEEDS THIS ORDER BECAUSE IT USES
# MESH AND MODEL
from . import obj
```



























