import bpy, os
import numpy as np
from bpy.props import *
from bpy.types import Panel, Operator, Menu
from bpy.utils import previews

# Info
bl_info = {
    "name": "QuickNormal",
    "author": "Christian Walters",
    "description": "Quick and dirty image to normal map converter",
    "version": (0, 7, 1),
    "blender": (2, 80, 0),
    "location": "Material Properties",
    "category": "Material",
}

# Add-on path
addon_dir = os.path.dirname(__file__)

class QuickNormalMapPanel(bpy.types.Panel):
    # Creates a Panel in the Material context
    bl_label = "Quick Normal Map Generator"
    bl_idname = "MATERIAL_PT_quick_normal_map_generator"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        
        col = layout.column(align=True)
        col.label(text="Select an image")
        col.template_ID_preview(scene, "quick_normal_map_image", open="image.open")
        
        col.operator("quick.normal_generate", text="Generate Normal Map")

# Manual implementation of scipy's convolve2d() function
def convolve2d_manual(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='wrap')
    output = np.zeros_like(image)
    
    # Perform convolution
    for row in range(image_height):
        for col in range(image_width):
            region = padded_image[row:row + kernel_height, col:col + kernel_width]
            output[row, col] = np.sum(region * kernel)
            
    return output

def apply_sobel_filter(grayscale_image):
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply Sobel filter
    grad_x = convolve2d_manual(grayscale_image, sobel_x)
    grad_y = convolve2d_manual(grayscale_image, sobel_y)
    
    normals = np.zeros(grad_x.shape + (3,))
    normals[..., 0] = grad_x
    normals[..., 1] = grad_y
    normals[..., 2] = 1
    
    norm_vectors = np.sqrt(normals[..., 0]**2 + normals[..., 1]**2 + normals[..., 2]**2)
    normals[..., 0] /= norm_vectors
    normals[..., 1] /= norm_vectors
    normals[..., 2] /= norm_vectors
    
    normal_map = normals * 0.5 + 0.5
    
    return normal_map
    
class QuickNormalMapOperator(bpy.types.Operator):
    # Generate a normal map from an image
    bl_idname = "quick.normal_generate"
    bl_label = "Generate Normal Map"

    @classmethod
    def poll(cls, context):
        return context.scene.quick_normal_map_image is not None

    def execute(self, context):
        original_image = context.scene.quick_normal_map_image
        if original_image is None:
            self.report({'ERROR'}, "No image selected")
            return {'CANCELLED'}
        
        # Reshape pixels into 2D numpy array of grayscale values
        pixels = np.array(original_image.pixels[:])
        rgba_pixels = pixels.reshape((original_image.size[1], original_image.size[0], 4))
        weights = np.array([0.299, 0.587, 0.114])
        grayscale_pixels = np.dot(rgba_pixels[..., :3], weights)
        
        # Normalize pixels and apply sobel filter
        grayscale_pixels_normalized = grayscale_pixels / np.max(grayscale_pixels)
        grayscale2d = grayscale_pixels_normalized.reshape((original_image.size[1], original_image.size[0]))
        normal_map_data = apply_sobel_filter(grayscale2d)
        
        # New normal map image
        normal_map_rgba = np.zeros((original_image.size[1], original_image.size[0], 4))
        normal_map_rgba[..., :3] = normal_map_data
        normal_map_rgba[..., 3] = 1.0
        normal_map_flattened = normal_map_rgba.flatten()
        
        file_name = os.path.splitext(original_image.name)[0]
        file_extension = os.path.splitext(original_image.name)[1]
        new_name = f"{file_name}-normalMap{file_extension}"
        width, height = original_image.size
        normal_map_image = bpy.data.images.new(name=new_name, width=width, height=height, alpha=True)
        normal_map_image.pixels = list(normal_map_flattened)
        
        normal_map_image.update()
        context.scene.quick_normal_map_image = normal_map_image
        
        # Save normal map image to the local hard drive
        if original_image.filepath_raw:
            base_path = os.path.dirname(bpy.path.abspath(original_image.filepath_raw))
        else:
            base_path = bpy.app.tempdir

        save_path = os.path.join(base_path, new_name)

        normal_map_image.filepath_raw = save_path
        normal_map_image.file_format = file_extension[1:].upper()
        normal_map_image.save()

        # Save and pack image to blend file     
        #normal_map_image.pack() 
        self.report({'INFO'}, f"Normal map: {normal_map_image.name}")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(QuickNormalMapPanel)
    bpy.utils.register_class(QuickNormalMapOperator)
    bpy.types.Scene.quick_normal_map_image = bpy.props.PointerProperty(
        name="Image",
        type=bpy.types.Image,
        description="Select a base image from which to generate a normal map."
    )

def unregister():
    bpy.utils.unregister_class(QuickNormalMapPanel)
    bpy.utils.unregister_class(QuickNormalMapOperator)
    del bpy.types.Scene.quick_normal_map_image

if __name__ == "__main__":
    register()