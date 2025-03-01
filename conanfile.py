import os

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain


class Recipe(ConanFile):
    settings = "os", "arch", "compiler", "build_type"

    def requirements(self):
        self.requires("gsl-lite/0.42.0")
        self.requires("fmt/11.1.1")

    def build_requirements(self):
        self.tool_requires("cmake/3.31.5")
        self.tool_requires("ninja/1.12.1")

    def layout(self):
        compiler = str(self.settings.compiler).lower()
        build_type = str(self.settings.build_type).lower()
        self.folders.build = f"build/{compiler}-{build_type}"
        self.folders.generators = os.path.join(self.folders.build, "generators")
        self.folders.build_folder_vars = [
            "settings.compiler",
            "settings.build_type",
        ]

    def generate(self):
        deps = CMakeDeps(self)
        deps.check_components_exist = True
        deps.generate()
        tc = CMakeToolchain(self)
        tc.user_presets_path = "CMakeConanPresets.json"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
