#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Vertices coordinates
GLfloat vertices[] = {
  -0.5f, -0.5f * float(std::sqrt(3)) / 3    , 0.0f, // Lower left corner
   0.5f, -0.5f * float(std::sqrt(3)) / 3    , 0.0f, // Lower right corner
   0.0f,  0.5f * float(std::sqrt(3)) * 2 / 3, 0.0f  // Upper corner
};

std::string getFileContent(const char* filename)
{
  std::ifstream file(filename);

  if (file.is_open())
  {
    std::string content((std::istreambuf_iterator<char>(file)),
                        (std::istreambuf_iterator<char>()));

    file.close();

    return content;
  } else {
    std::cout << "Failed to open the file " << filename << std::endl;
    exit(1);
  }
}

int main()
{
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  // Creating a glfw window    
  GLFWwindow* glfwWindow = glfwCreateWindow(1280, 720, "Test OpenGL", NULL, NULL);

  if (glfwWindow == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    exit(1);
  }
  else
  {
    std::cout << "GLFW window created" << std::endl;
  }

  glfwMakeContextCurrent(glfwWindow);

  // Loading all OpenGL function pointers with glad
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    exit(1);
  }
  else
  {
    std::cout << "GLAD initialized successfully" << std::endl;
  }

  // Shaders
  std::string vertexShaderCode = R"(
  #version 330 core
  in vec3 position;

  void main() {
      gl_Position = vec4(position, 1.0f);
  }
  )";
  
  std::string fragmentShaderCode = R"(
  #version 330 core
  out vec4 outColor;

  void main() {
      outColor = vec4(0.6f, 0.3f, 0.0f, 1.0f);
  }
  )";

  const char* vertexShaderSource = vertexShaderCode.c_str();
	const char* fragmentShaderSource = fragmentShaderCode.c_str();

	// Create Vertex Shader Object and get its reference
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// Attach Vertex Shader source to the Vertex Shader Object
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	// Compile the Vertex Shader into machine code
	glCompileShader(vertexShader);

	// Create Fragment Shader Object and get its reference
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	// Attach Fragment Shader source to the Vertex Shader Object
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	// Compile the Fragment Shader into machine code
	glCompileShader(fragmentShader);

	// Create Shader Program Object and get its reference
	GLuint shaderProgram = glCreateProgram();
	// Attach the Vertex and Fragment Shaders to the Shader Program
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	// Wrap-up / link all the shaders together into the Shader Program
	glLinkProgram(shaderProgram);

	// Delete the now useless Vertex and Fragment Shader Objects
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

  // Use the shader program
  glUseProgram(shaderProgram);

  // Create reference containers for the Vertex Array Object and the Vertex Buffer Object
	GLuint VAO, VBO;

	// Generate the VAO and VBO with only 1 object each
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Make the VAO the current Vertex Array Object by binding it
	glBindVertexArray(VAO);

	// Bind the VBO specifying it's a GL_ARRAY_BUFFER
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// Introduce the vertices into the VBO
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Configure the Vertex Attribute so that OpenGL knows how to read the VBO
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	// Enable the Vertex Attribute so that OpenGL knows to use it
	glEnableVertexAttribArray(0);

	// Bind both the VBO and VAO to 0 so that we don't accidentally modify the VAO and VBO
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Specify the color of the background
	glClearColor(0.27f, 0.13f, 0.17f, 1.0f);
	// Clean the back buffer and assing the new color to it
	glClear(GL_COLOR_BUFFER_BIT);
	// Swap the back buffer with the front buffer
	glfwSwapBuffers(glfwWindow);


  std::cout << "Opening GLFW window" << std::endl;

  while (!glfwWindowShouldClose(glfwWindow))
  {
    // Using GLFW to check and process input events internally
    glfwPollEvents();
	
		glClear(GL_COLOR_BUFFER_BIT);

		// Draw the triangle using the GL_TRIANGLES primitive
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glfwSwapBuffers(glfwWindow);
  }
  
  std::cout << "GLFW window closed" << std::endl;

  return 0;
}