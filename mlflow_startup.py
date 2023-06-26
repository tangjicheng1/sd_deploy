from webui_wrapper import StableDiffusionWebuiWrapper
from webui_wrapper import decode_base64_to_image
from modules.api import models

import mlflow

# Define custom Python model class
class StableDiffusion(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    # mount S3
    self.webui_wrapper = StableDiffusionWebuiWrapper()
    self.webui_wrapper.initialize()

  def predict(self, context, input):
    output = self.webui_wrapper.txt2img(input['model_name'][0], input['model_input'][0])
    return output

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, TensorSpec

# Define input and output schema
input_schema = Schema([ColSpec(DataType.string, "model_name"), ColSpec(DataType.string, "model_input"),])
output_schema = Schema([ColSpec(DataType.string, "output")])
signature = ModelSignature(inputs=input_schema,outputs=output_schema)

# Define input example
input_example= {'model_name' : "MeinaMix.safetensors" , 'model_input' : '''{
                    "model_name": "MeinaMix.safetensors",
                    "enable_hr": false,
                    "denoising_strength": 0.5,
                    "firstphase_width": 0,
                    "firstphase_height": 0,
                    "hr_scale": 2,
                    "hr_upscaler": "string",
                    "hr_second_pass_steps": 0,
                    "hr_resize_x": 0,
                    "hr_resize_y": 0,
                    "hr_sampler_name": "string",
                    "hr_prompt": "",
                    "hr_negative_prompt": "",
                    "prompt": "1girl",
                    "styles": [
                        "string"
                    ],
                    "seed": 123,
                    "subseed": 123,
                    "subseed_strength": 0,
                    "seed_resize_from_h": -1,
                    "seed_resize_from_w": -1,
                    "sampler_name": "LMS",
                    "batch_size": 1,
                    "n_iter": 1,
                    "steps": 20,
                    "cfg_scale": 7,
                    "width": 512,
                    "height": 512,
                    "restore_faces": false,
                    "tiling": false,
                    "do_not_save_samples": false,
                    "do_not_save_grid": false,
                    "negative_prompt": "",
                    "eta": 0,
                    "s_min_uncond": 0,
                    "s_churn": 0,
                    "s_tmax": 0,
                    "s_tmin": 0,
                    "s_noise": 1,
                    "override_settings": {},
                    "override_settings_restore_afterwards": true,
                    "script_args": [],
                    "sampler_index": "Euler",
                    "script_name": "",
                    "send_images": true,
                    "save_images": false,
                    "alwayson_scripts": {}
                    }'''}

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=StableDiffusion(),
        artifacts={},
        pip_requirements=["transformers","torch", "accelerate", "xformers"],
        input_example=input_example,
        signature=signature
    )

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "stable-diffusion-webui-1"
)

# Load the logged model
loaded_model = mlflow.pyfunc.load_model('runs:/'+run.info.run_id+'/model')

# Make a prediction using the loaded model
result_image = loaded_model.predict(input_example)
pic = decode_base64_to_image(result_image)
pic.save("mlflow_1.jpg")
