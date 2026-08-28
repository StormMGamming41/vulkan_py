from src import vulkan_py as vk
from ctypes import c_uint32, c_uint64, c_void_p, c_float, byref, pointer, cast, POINTER, c_char_p
import ctypes
import glfw
import os

def check(result, what):
    if result != vk.VkResult.VK_SUCCESS:
        print(f"{what} failed: VkResult={result}")

ENABLE_VALIDATION_LAYERS = True

class App:

    def __init__(self, name='test', res=[800,600]):

        self.name = name
        self.res = res

    def _debug_callback(self, severity, msg_type, callback_data_ptr, user_data):
            data = callback_data_ptr.contents
            message = data.pMessage.decode(errors="ignore") if data.pMessage else "<no message>"
    
            if severity >= vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
                tag = "ERROR"
            elif severity >= vk.VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
                tag = "WARN"
            else:
                tag = "INFO"
    
            print(f"[validation:{tag}] {message}")
            return 0   # VK_FALSE - don't abort the call that triggered this

    def create_window(self):

        if not glfw.init():
            raise RuntimeError("glfw initialization failed")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(self.res[0], self.res[1], self.name, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("failed to create glfw window")

    def create_instance(self):

        app_info = vk.VkApplicationInfo(
            pApplicationName=self.name.encode('utf-8'),
            applicationVersion=1,
            pEngineName=b'vukan_py',
            engineVersion=1,
            apiVersion=(1 << 22)
        )

        exts = glfw.get_required_instance_extensions()
        ext_bytes = [ext.encode('utf-8') for ext in exts]
        ext_arr = (c_char_p * len(ext_bytes))(*ext_bytes)

        layers = []
        if ENABLE_VALIDATION_LAYERS:
            layers = [b"VK_LAYER_KHRONOS_validation"]
        layer_arr = (c_char_p * len(layers))(*layers)

        create_info = vk.VkInstanceCreateInfo(
            applicationInfo=app_info,
            enabledExtensionCount=len(ext_bytes),
            ppEnabledExtensionNames=cast(ext_arr, POINTER(c_char_p)),
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=cast(layer_arr, POINTER(c_char_p)),
        )

        raw_instance = vk.VkInstance()
        result = vk.vkCreateInstance(create_info, None, raw_instance)
        if result != vk.VkResult.VK_SUCCESS:
            raise RuntimeError(f"failed to create vulkan instance: {result}")
        self.raw_instance = raw_instance
        self.instance = vk.Instance(raw_instance)

    def create_surface(self):

        self.surface_handle = c_void_p(0)
        result = glfw.create_window_surface(self.raw_instance.value, self.window, None, byref(self.surface_handle))
        if result != 0:
            raise RuntimeError(f"glfw surface creation failed: {result}")
        self.surface = vk.VkSurfaceKHR(self.surface_handle.value)

    def enumerate_physical_devices(self):

        count = c_uint32()
        result = self.instance.vkEnumeratePhysicalDevices(byref(count), None)
        if result != vk.VkResult.VK_SUCCESS:
            raise RuntimeError(f"physical device enumeration failed: {result}")
        if count.value == 0:
            raise RuntimeError("vulkan compatible hardware not found")

        self.physical_devices = (vk.VkPhysicalDevice * count.value)()
        self.instance.vkEnumeratePhysicalDevices(byref(count), self.physical_devices)

    def find_queue_families(self, physical_device):

        queue_count = c_uint32()
        self.instance.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, byref(queue_count), None)
        queue_families = (vk.VkQueueFamilyProperties * queue_count.value)()
        self.instance.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, byref(queue_count), queue_families)

        graphic_family = present_family = None
        for i, fam in enumerate(queue_families):
            if fam.queueFlags & vk.VkQueueFlagBits.VK_QUEUE_GRAPHICS_BIT:
                graphic_family = i
            supported = c_uint32(0)
            self.instance.vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, self.surface, byref(supported))
            if supported.value:
                present_family = i
            if graphic_family is not None and present_family is not None:
                break
        return graphic_family, present_family

    def select_physical_device(self):

        best = None

        for pd in self.physical_devices:
            gfx, prf = self.find_queue_families(pd)
            if gfx is None or prf is None:
                continue
            props = vk.VkPhysicalDeviceProperties()
            self.instance.vkGetPhysicalDeviceProperties(pd, props)
            score = 2 if props.deviceType == vk.VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 1
            if best is None or score > best[0]:
                best = (score, pd, gfx, prf, props.deviceName.decode(errors="ignore"))

            if best is None:
                raise RuntimeError("Compatible GPU with graphics and present support not found")

            _, pd, gfx, prf, name = best
            self.physical_device, self.graphics_family, self.present_family = pd, gfx, prf
            print(f"Using GPU: {name} (graphics family={gfx}, present family={prf})") 

    def create_logical_device(self):

        unique_families = {self.graphics_family, self.present_family}
        priority = c_float(1.0)
        queue_infos = [vk.VkDeviceQueueCreateInfo(queueFamilyIndex=fam, queueCount=1, pQueuePriorities=pointer(priority)) for fam in unique_families]
        queue_infos_arr = (vk.VkDeviceQueueCreateInfo * len(queue_infos))(*queue_infos)

        device_ext = [b'VK_KHR_swapchain']
        ext_arr = (c_char_p * len(device_ext))(*device_ext)
        features = vk.VkPhysicalDeviceFeatures()

        create_info = vk.VkDeviceCreateInfo(
            queueCreateInfoCount=len(queue_infos),
            pQueueCreateInfos=cast(queue_infos_arr, POINTER(vk.VkDeviceQueueCreateInfo)),
            enabledExtensionCount=len(device_ext),
            ppEnabledExtensionNames=cast(ext_arr, POINTER(c_char_p)),
            pEnabledFeatures=pointer(features)
        )

        raw_device = vk.VkDevice()
        check(self.instance.vkCreateDevice(self.physical_device, create_info, None, raw_device), "vkCreateDevice")
        self.raw_device = raw_device
        self.device = vk.Device(raw_device)
        
        graphics_queue = vk.VkQueue()
        self.device.vkGetDeviceQueue(self.graphics_family, 0, byref(graphics_queue))
        self.graphics_queue = graphics_queue
        
        present_queue = vk.VkQueue()
        self.device.vkGetDeviceQueue(self.present_family, 0, byref(present_queue))
        self.present_queue = present_queue

    def main_loop(self):

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

    def run(self):

        self.create_window()
        self.create_instance()
        self.create_surface()
        self.enumerate_physical_devices()
        self.select_physical_device()
        self.create_logical_device()
        try:
            self.main_loop()
        finally:
            self.cleanup()

    def cleanup(self):

        self.device.vkDestroyDevice(None)
        self.instance.vkDestroySurfaceKHR(self.surface, None)
        self.instance.vkDestroyInstance(None)

        glfw.destroy_window(self.window)
        glfw.terminate()


App().run()
