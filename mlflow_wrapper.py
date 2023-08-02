import base64
from io import BytesIO

import mlflow
from PIL import Image
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, TensorSpec

import webui_wrapper

mlflow.set_experiment("/Users/tang.j@ctw.inc/sd0728")

# Define custom Python model class
class StableDiffusion(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    pass

  def predict(self, context, input):
    output = webui_wrapper.simple_txt2img_test(input["input"][0])
    return output


# Define input and output schema
input_schema = Schema([ColSpec(DataType.string, "input"),])
output_schema = Schema([ColSpec(DataType.string, "output")])
signature = ModelSignature(inputs=input_schema,outputs=output_schema)

# Define input example
input_value = '''{
                "sd_model_checkpoint": "v1-5-pruned-emaonly",
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
                }'''

input_example = {"input": input_value}

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=StableDiffusion(),
        artifacts={},
        pip_requirements=["transformers","torch", "torchvision", "accelerate", "xformers","piexif", "gradio"],
        input_example=input_example,
        signature=signature,
        code_path=["/Workspace/Repos/tang.j@ctw.inc/sd_deploy"],
        await_registration_for=6000,
    )

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "func0728",
    await_registration_for=6000,
)

# Load the logged model
loaded_model = mlflow.pyfunc.load_model('runs:/'+run.info.run_id+'/model')

# Make a prediction using the loaded model
result_image = loaded_model.predict(input_example)
pic = Image.open(BytesIO(base64.b64decode(result_image)))
pic.save("mlflow_1.jpg")
