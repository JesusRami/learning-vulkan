//#define NDEBUG
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <string.h>

const int WIDTH = 800;
const int HEIGHT = 600;

const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif


class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow *window;
	//Connection between the application and Vulkan library
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;

	void initWindow()
	{
		//Initialize library
		glfwInit();

		//Specify don't create OpenGL context
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(WIDTH,HEIGHT,"Vulkan",nullptr,nullptr);
	}
	
	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
	}

	void mainLoop()
	{
		while(!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
		}
	}
	
	void cleanup()
	{
		if(enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void createInstance()
	{
		VkApplicationInfo appInfo = {};
		VkInstanceCreateInfo createInfo = {};
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
		uint32_t extensionCount;
		
		//Check availability of required validation layers
		if(enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but no available!");
		}

		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		//Get required extensions
		auto requiredExtensions = getRequiredExtensions();

		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
		createInfo.ppEnabledExtensionNames = requiredExtensions.data();
		if (enableValidationLayers) 
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}
		

		//Get provided extensions
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> providedExtensions(extensionCount);
		
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, providedExtensions.data());

		//Check that the required extensions are provided
		for(int i = 0; i < static_cast<uint32_t>(requiredExtensions.size()); i++)
		{
			bool found = false;
			for(const auto& extension : providedExtensions)
			{
				if(strcmp(extension.extensionName,requiredExtensions[i]) == 0)
				{
					found = true;
					std::cout<<"\t" <<extension.extensionName<<" found"<<std::endl;
					break;
				}
			}
			if(!found)
			{
				throw std::runtime_error("Extension not found");
			}
		}



		
		if(vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create instance");
		}


	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount = 0;
		bool layerFound;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		
		std::vector<VkLayerProperties> availableLayers(layerCount);

		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for(const char * layerName : validationLayers)
		{
			layerFound = false;
			for (const auto& layerProperties: availableLayers)
			{
				if(strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if(!layerFound)
			{
				return false;
			}
		}
		return true;
	}

	std::vector<const char *> getRequiredExtensions() 
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions,glfwExtensions + glfwExtensionCount);

		if(enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr<<"validation layer: " << pCallbackData->pMessage<<std::endl;

		return VK_FALSE;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}
	void setupDebugMessenger()
	{
		VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
		if(!enableValidationLayers) return;

		populateDebugMessengerCreateInfo(createInfo);		

		if(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to set up debug messenger");
		}
	}

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
			const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
			const VkAllocationCallbacks* pAllocator,
			VkDebugUtilsMessengerEXT* pDebugMessenger)
	{
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
		if(func != nullptr)
		{
			return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
		}
		else
		{
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	static void DestroyDebugUtilsMessengerEXT(VkInstance instance,
		VkDebugUtilsMessengerEXT debugMessenger,
		const VkAllocationCallbacks* pAllocator)
	{
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT");
		if (func != nullptr)
		{
			func(instance, debugMessenger, pAllocator);
		}
	}


};

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch(const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}