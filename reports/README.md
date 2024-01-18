---
layout: default
nav_exclude: true
---

# AvianSight: Bird Species Classification for Amateur Birdwatchers

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [x] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [x] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on learn.inside.dtu.dk**
>
> Answer:

This project has been executed by group: MLOps 4.

### Question 2
> **Enter the study number for each member in the group**
>
> Answer:

This project has been executed by students: s184202, s193973, s222681 and s222698

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer:

We chose to work with the TIMM (PyTorch Image Models), PyTorch Lightning and TorchMetrics frameworks; where TIMM's pre-trained models, PyTorch Lightning's training framework and TorchMetrics' performance metrics formed a robust environment for developing a bird species classifier.

The TIMM framework was central to our model development with its wide range of pre-trained computer vision models which allowed leveraging transfer learning in fine-tuning a classification model. The function `create_model` was used to initialise model architectures with pre-trained weights and tailor them to our dataset.

A training pipeline was set up with PyTorch Lightning’s LightningModule and Trainer; organising training and validation steps, model checkpoints and logging with WANDB. This enabled us to focus on the model's logic rather than the boilerplate code. TorchMetrics was similarly used to handle performance metrics, providing a convenient and reliable way to calculate accuracy to be logged during both training and validation phases.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

--- question 4 fill here ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- question 5 fill here ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- question 6 fill here ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**

In total, we have implemented four tests. These tests have been made to guarantee the reliability and robustness of our data 
handling and model generation components. For the data handling a single test was created, `test_data_loading`, that assures that 
the data is properly loaded. 

For the model generation, we implemented three tests: `test_forward_pass` - which tests the forward pass of the model, 
`test_training_step` - which tests the training pass of the model, and `test_validation_step` - which tests the validation pass of  
the model.
	

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**

The total code coverage of the code is 90%, as can be seen from the output of `coverage report`:
```
Name                     Stmts   Miss  Cover
--------------------------------------------
src/__init__.py              0      0   100%
src/data/__init__.py         0      0   100%
src/data/data.py            32      4    88%
src/models/__init__.py       0      0   100%
src/models/model.py         50     14    72%
tests/__init__.py            4      0   100%
tests/test_data.py          55      0   100%
tests/test_model.py         40      0   100%
--------------------------------------------
TOTAL                      181     18    90%

```
Even if the coverage is 10% far from the perfect one, this does not guarantee the lack of bugs and errors in the code.
Despite our extensive testing of the data loader and model training modules, unforeseen interactions and edge cases may still exist. 
Nevertheless, a high code coverage is a good indicator that the code has been tested.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**

We made use of both branches and PRs in our project. For every task, we created a branch and also protected the main branch by
adding the following rules: at least one person needs to approve any PR, all your workflows have to pass and all conversations need 
to be resolved. By using branches and pull requests in version control it ensures that changes are reviewed before merging,
maintaining code integrity, and facilitating smoother project evolution.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- question 10 fill here ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**

We set up our Continuous Integration (CI) using two separate files: one for checking code standards, named .github/workflows/codecheck.yml, and another for running unit tests, called .github/workflows/tests.yml. To make sure our code meets standards, we use tools like Ruff and MyPy for type checking. We only tested the code on one operating system, specifically ubuntu-20.04, as it was the operating system used for developing the project, and on one version of Python, namely python 3.10.0, as required by the project.

To make the processes faster, we use a caching mechanism. This way, every package we download won't be deleted after the workflow finishes, improving the overall speed of the workflow. Additionally, we included a `requirements_tests.txt` file with the specific packages required for running the workflow - for example, typing packages required by mypy.

Our GitHub Actions workflows run automatically every time we merge a branch into the main branch or create a pull request. This helps us ensure our project's integrity and quality by consistently checking for issues and making sure everything works well.

An example of a triggered workflow can be seen here: <https://github.com/CristianaLazar/mlops-bird-classification-project/actions/runs/7555347779/job/20570134093>

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- question 12 fill here ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- question 13 fill here ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>

As reproducibility is crucial, for this project, we developed several images: one for training, one for inference and one for deployment - to guarantee that the application can run on any device.  The following commands can be used to create and run the docker files:

To build the docker file into a docker image:
```
    docker build -f dockerfiles/trainer.dockerfile . -t trainer:latest
    docker build -f dockerfiles/predicter.dockerfile . -t predicter:latest
```

To run the docker images:
```
    docker run --name experiment1 trainer:latest
    docker run --name predict predicter:latest 
```

To automate the process even more, we created in Google Cloud a trigger for docker image creation. Every time a branch is merged into `main`, the docker files are created by using the configurations from `cloudbuild.yaml`. Once constructed, these docker images are executed using Google Cloud. 

A link to the training Docker file can be found [here](https://github.com/CristianaLazar/mlops-bird-classification-project/blob/main/dockerfiles/trainer.dockerfile) 





### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

During our bird classification project, we adopted a systematic approach for debugging. The primary tool for this was the Python debugger, which allowed us to set breakpoints in the code. In our development environment, particularly VS Code, we used the F9 key to insert these inline breakpoints, visible as small red dots next to the code lines. This feature enabled us to execute the script in debug mode and step through the code interactively, observing the behavior and state of variables at each step. This method proved invaluable in identifying and fixing bugs efficiently.

In addition to traditional debugging, we utilized PyTorch Lightning's simple profiler. This profiler is specifically designed for deep learning tasks and profiles key actions in the training loop, including `on_epoch_start`, `on_epoch_end`, `on_batch_start`, `tbptt_split_batch`, `model_forward`, `model_backward`, `on_after_backward`, `optimizer_step`, `on_batch_end`, `training_step_end`, and `on_training_end`. This comprehensive profiling helped us understand the performance of different segments of our code during the training process.
After deploying our model on the school's high-performance computing (HPC) resources, we analyzed the profiler's output and found no significant bottlenecks.


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

In our bird classification project, we leveraged various Google Cloud Platform (GCP) services, each serving a distinct role:

1. **Vertex AI API**: This service was crucial for building, deploying, and scaling our machine learning models. We primarily used it for training our bird classification model.

2. **Artifact Registry**: This registry was a key component in our CI/CD pipeline, managing our Docker images which were essential for automating the deployment process.

3. **Google Cloud Storage (Bucket)**: Served as our primary object storage solution, where we stored large datasets of bird images and other related data.

4. **Cloud Logging**: This was integral for aggregating logs from different services and VMs, aiding significantly in monitoring application activities and debugging.

5. **Cloud Monitoring**: Provided real-time metrics, dashboards, and alerts, which were vital in tracking our application's performance and health.

6. **Container Registry**: Initially used for hosting our Docker container images, this service facilitated easy management and deployment of these images.

7. **Compute Engine**: The backbone of our application, offering scalable virtual machines (VMs) for processing tasks and running backend services.

8. **Cloud Build**: Automated our build, test, and deployment processes, enhancing our development workflow's efficiency.

9. **Cloud Run**: Enabled us to deploy and manage containerized applications seamlessly on a fully managed serverless platform.

10. **Cloud Triggers (Cloud Functions)**: Used for automatically initiating processes or workflows, model training in response to specific changes to main branch.

11. **Identity and Access Management (IAM)**: Managed user access and permissions, ensuring secure, controlled access to our GCP resources.

12. **Google Cloud Console**: Offered a comprehensive, user-friendly interface for managing and monitoring all our GCP resources, services, and applications.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

In our project, we initially faced challenges in being granted GPU resources on Google Cloud Platform (GCP). Consequently, our initial setup involved running a custom Vertex AI job, where the backbone was the Compute Engine's 'n1-highmem-2' instance. This configuration provided us with high memory capacity, crucial for our data-intensive tasks, but lacked GPU acceleration.

After reaching out for support and being granted access to GPU resources, we shifted to a more robust setup. We created a new job that leveraged the NVIDIA T4 GPU, a significant upgrade for our computational needs. This GPU-enabled instance allowed us to accelerate our model training and inference processes significantly.

For both setups, we utilized Docker containers. These containers were pre-loaded with all the necessary dependencies and our application code. This approach ensured a consistent and reproducible environment across different stages of our project. 


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
>
> Answer:

Buckets created in the project:
![Buckets](figures/buckets.png)

Example of structure in a data bucket:
![Content example of Bucket](figures/bucket_content.png)


### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
>
> Answer:

Project container registry:
![Containers](figures/container_reg.png)


### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer:

The deployment process involved wrapping our trained model within a FastAPI application that enables users to run inference on JPG images of birds to receive the classification bird species name and certainty/probability as response. The application was then containerised to ensure consistent runs across environments by packaging the application and its dependencies into a Docker image. After verifying that the image ran as intended locally, it was pushed to the project's Cloud Registry and deployed with Cloud Run.

To invoke the deployed service, users can send a POST request to the inference endpoint with a JPG image, replacing [path/to/image.jpg] with the correct image path:
curl -X POST -F "bird_image=@[path/to/image.jpg]" https://aviansight-app-5gmfsq67rq-ew.a.run.app/infer_image

PS: POST a selfie and see if you discover the application's Looney Bird easter egg.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Yes, we successfully implemented monitoring for our deployed model. Utilizing the Monitoring tab in our Cloud Run service, we were able to access diverse plots displaying key performance metrics. This setup allowed us to continuously observe and evaluate the operational aspects of our application, ensuring optimal performance and rapid response to any issues.

Additionally, we established a Service Level Objective (SLO) focusing on critical parameters such as availability, latency, and CPU usage time. These objectives serve as benchmarks against which we can measure the reliability and efficiency of our service, providing a structured approach to maintaining high service quality.

To further enhance our monitoring capabilities, we set up several alerts within the monitoring service. These alerts track metrics like the sum of cloud function invocations, CPU utilization, and ingress bytes. By doing so, we are promptly notified of significant changes or potential issues, such as unusual spikes in function calls, high CPU demand, or abnormal network traffic. This proactive monitoring strategy is pivotal in ensuring the longevity of our application, as it enables us to swiftly identify and rectify operational inefficiencies or disruptions, thereby maintaining consistent service quality.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---