VULKAN_SDK_PATH = /home/jesus/VULKAN/1.1.106.0/x86_64
CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

VulkanTest: first.cpp
	g++ $(CFLAGS) -o VulkanTest first.cpp $(LDFLAGS)

.PHONY: test clean
	
test: VulkanTest
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/explicit_layer.d
	./VulkanTest
clean:
	rm -f VulkanTest