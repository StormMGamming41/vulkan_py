# Vulkan Python Point-Light Demo

This is a complete small renderer built around the `Vulkan-vk-xml-to-py-parser` binding.

The binding is expected to be available exactly like the repository's current example:

```python
from src import vulkan_py as output
```

The renderer contains:

- Vulkan instance/device/swapchain
- GLFW window and Vulkan surface
- depth buffer
- indexed cube geometry
- floor geometry
- uniform buffer
- descriptor set
- push constants
- four point lights
- Lambert diffuse lighting
- Blinn-Phong specular lighting
- distance attenuation
- animated cubes
- Vulkan validation callback when `VK_EXT_debug_utils` and the validation layer are installed

## Files

```text
lighting_demo.py
shaders/
    lighting.vert
    lighting.frag
    lighting.vert.spv
    lighting.frag.spv
build_shaders.bat
```

## Put it in the binding repository

The intended layout is:

```text
Vulkan-vk-xml-to-py-parser/
├── src/
│   └── vulkan_py/
├── lighting_demo.py
├── build_shaders.bat
└── shaders/
```

## Dependencies

Python:

```text
glfw
```

System:

```text
Vulkan loader/driver
Vulkan SDK
glslc
```

Build shaders:

```text
build_shaders.bat
```

Then:

```text
python lighting_demo.py
```

## Lighting

The fragment shader calculates:

```text
ambient
+
Σ(diffuse × attenuation)
+
Σ(specular × attenuation)
```

Each point light is:

```text
position.xyz
radius/intensity.w

color.rgb
intensity.w
```

The four lights are deliberately placed around the scene so the diffuse and specular components are obvious.

## Important implementation detail

The descriptor UBO is deliberately shared by all objects in a frame. The model matrix is supplied through a push constant, so each object can have its own transform without allocating a separate descriptor set or uniform buffer.

The UBO is:

```text
view
projection
camera position
4 light positions
4 light colors
```

The push constant is:

```text
model matrix
```

## Notes

This version intentionally keeps swapchain resize disabled. That removes a large amount of unrelated swapchain-recreation code from the lighting example.

The renderer uses host-visible, host-coherent buffers for geometry and uniforms. This is not the final performance-oriented design, but it makes the example substantially smaller and easier to verify.

Once this works, the next useful upgrade is:

```text
staging buffers
→ device-local vertex/index buffers
→ dynamic camera
→ configurable lights
→ textures/materials
```
