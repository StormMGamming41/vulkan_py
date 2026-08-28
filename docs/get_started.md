# Getting Started

This guide covers the basic setup required to start using the generated Vulkan Python binding.

## Requirements

Before using the binding, make sure you have the following installed.

### Python

Python 3.x is required.

You can check your installation with:

```bash
python --version
```

### Vulkan Runtime

You need a working Vulkan-capable GPU and Vulkan driver.

Verify that Vulkan is available on your system using the Vulkan tools provided with your driver or SDK.

### Vulkan SDK

The Vulkan SDK is recommended for development.
It provides useful development tools such as:

- Vulkan headers and registry files
- Validation layers
- Vulkan loader/development components
- Shader compilation tools
- Vulkan utilities

#### Download the Vulkan SDK from:

https://vulkan.lunarg.com/

After installing it, make sure the SDK environment is configured correctly.

You can verify the installation in terminal with:
```bash
vulkaninfo
```

If vulkaninfo successfully reports your Vulkan devices and capabilities, your Vulkan development environment is ready.

### GLFW

The graphical examples use GLFW for window creation.

Install it with:
```bash
pip install glfw
Getting the Project
```

### The ```vulkan_py``` Package

Install the python package with:
```bash
pip install vulkan_py
```

During development, the package can be imported with:
```python
import vulkan_py
```
Or for better convenience:
```python
from vulkan_py import *
```
At this point, your environment is ready to begin development.

### First Test

Create a python file, name it anything you like
```python
from src import vulkan_py as vk

print("Vulkan Python binding imported successfully.")
```

Run it:
```bash
python test.py
```
If it runs without an import error, the Python side of the setup is working.

## Vulkan Development Environment

For actual Vulkan development, it is strongly recommended to enable the Vulkan validation layers provided by the Vulkan SDK.

Validation layers can detect problems such as:

Invalid Vulkan API usage
Incorrect structure configuration
Synchronization errors
Invalid command usage
Resource lifetime problems
Swapchain mistakes

They are especially useful while developing applications with the binding.

## Where to Go Next

The basic environment is now ready.

Continue with the documentation in the docs/ directory:
```
docs/
├── getting_started.md
├── binding_usage.md
└── vulkan_guide.md
```
binding_usage.md covers how to work with the generated binding itself.

vulkan_guide.md takes you through Vulkan development step by step, starting from instance creation and progressing toward actual rendering.