# Run steps on dstack

[dstack](https://dstack.ai/) is an open source project, so you can use it for your own cloud accounts from AWS, GCP, Azure, Vast.ai, LambdaLabs, Datacrunch.io, TensorDock. However, I highly recommend to use [dstack Sky](https://sky.dstack.ai/) which is a cloud version of dstack server. As an organization, dstack provides GPU quotas through contracts with various cloud service providers. Hence, it is very likely you could get desired GPU machine. Plus, you will almost always be able to get spot instances too.

Below instructions assumes that you have dstack Sky account, but the same instructions could be applied if you have configured your own dstack server. If you are interested in to create dstack Sky account [check out this link](https://sky.dstack.ai).

## Install dstack client

```console
$ pip install dstack
```

## Config dstack client

For dstack Sky, the endpoint url should be `https://sky.dstack.ai`.

```console
$ dstack config \
--url <YOUR-DSTACK-SERVER-ENDPOINT> \
--project <YOUR-PROJECT-NAME> \
--token <YOUR-DSTACK-SERVER-TOKEN>
```

## Initialize dstack project

```console
$ dstack init
```

## Run a job

The following command launches a LLM fine-tuning job as specified in the [ft.dstack.yml](ft.dstack.yml). Or you could choose [batch_infer.dstack.yml] for batch inferencing step too.

```console
$ dstack run . -f ft.dstack.yml
```

When you hit the command, it will show you a list of plans that you can choose from. For instance, below shows a spot instance with 2 x A100(80GB) could be provisioned at $2.41615/h price in azure. 

```console
⠋ Getting run plan...
 Configuration          dstack.yaml                   
 Project                deep-diver-main               
 User                   deep-diver                    
 Pool name              default-pool                  
 Min resources          2..xCPU, 8GB.., 2xA100 (80GB) 
 Max price              -                             
 Max duration           72h                           
 Spot policy            auto                          
 Retry policy           no                            
 Creation policy        reuse-or-create               
 Termination policy     destroy-after-idle            
 Termination idle time  300s                          

 #  BACKEND  REGION        INSTANCE                  RESOURCES                     SPOT  PRICE      
 1  azure    westeurope    Standard_NC48ads_A100_v4  48xCPU, 440GB, 2xA100         yes   $2.41615   
                                                     (80GB), 100GB (disk)                           
 2  gcp      us-east4      a2-ultragpu-2g            24xCPU, 340GB, 2xA100         yes   $3.69698   
                                                     (80GB), 100GB (disk)                           
 3  gcp      europe-west4  a2-ultragpu-2g            24xCPU, 340GB, 2xA100         yes   $3.71127   
                                                     (80GB), 100GB (disk)                           
    ...                                                                                             
 Shown 3 of 10 offers, $11.3251 max
 Continue? [y/n]: 
```

If you say `y`, the job will be launched, and all the logs on the machine will be displayed until the job is finished. Or you can always go to the [dstack cloud dashboard](https://sky.dstack.ai/runs) to check the logs out at anytime.