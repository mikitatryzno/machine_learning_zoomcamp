### Building the Image

First, let's clone the course repository and build the Docker image:

```bash
# Clone the repository if you haven't already
git clone https://github.com/DataTalksClub/machine-learning-zoomcamp.git

# Navigate to the homework folder
cd machine-learning-zoomcamp/cohorts/2025/05-deployment/homework

# Build the Docker image
podman build -f Dockerfile_full -t zoomcamp-model:3.13.10-hw10 .
```

## Question 1: Testing the Model Locally

Let's run the Docker container and test it locally:

```bash
# Run the Docker container
podman run -it --rm -p 9696:9696 zoomcamp-model:3.13.10-hw10
```

In another terminal, let's run the test script:

```bash
python q6_test.py
```

The output is {'conversion_probability': 0.49999999999842815, 'conversion': False}

Probability of conversion is `0.499`


### Installing kubectl and kind

Let's make sure we have kubectl and kind installed:

```bash
# Check if kubectl is installed
kubectl version --client
# Install kubectl if needed (follow instructions from https://kubernetes.io/docs/tasks/tools/)
# Check if kind is installed
kind --version
# Install kind if needed (follow instructions from https://kind.sigs.k8s.io/docs/user/quick-start/)
```

## Question 2: Kind Version

The version of kind can be found by running:

```bash
kind --version
```

kind version is `0.30.0`.

### Creating a Cluster

Let's create a Kubernetes cluster using kind:

```bash
# Create a cluster
kind create cluster
# Check that the cluster was successfully created
kubectl cluster-info
```

## Question 3: Smallest Deployable Computing Unit

The smallest deployable computing unit in Kubernetes is a `Pod`.

## Question 4: Service Type

Let's check the services that are already running in the cluster:

```bash
kubectl get services
```

`ClusterIP` is the Type of the service that is already running.

## Question 5: Loading Docker Image to Kind

To use the Docker image we created (zoomcamp-model:3.13.10-hw10) with kind, we need to load it into the kind cluster:

```bash
kind load docker-image zoomcamp-model:3.13.10-hw10
```

This is the command we need to run, which is the answer to question.

## Question 6: Creating a Deployment

Let's create a deployment configuration file (deployment.yaml):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: subscription
spec:
  selector:
    matchLabels:
      app: subscription
  replicas: 1
  template:
    metadata:
      labels:
        app: subscription
    spec:
      containers:
      - name: subscription
        image: zoomcamp-model:3.13.10-hw10
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"            
          limits:
            memory: "128Mi"
            cpu: "200m"
        ports:
        - containerPort: 9696
```

The value for <Port> is 9696, which is the answer to question.

Let's apply this deployment:

```bash
kubectl apply -f deployment.yaml
# Check that the Pod is running
kubectl get pods
```

## Question 7: Creating a Service

Let's create a service configuration file (service.yaml):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: subscription
spec:
  type: LoadBalancer
  selector:
    app: subscription
  ports:
  - port: 80
    targetPort: 9696

```

Service is `subscription`, which is the answer to Question 7.

Let's apply this service:

```bash
kubectl apply -f service.yaml
# Check that the service is created
kubectl get services
```

### Testing the Service

Let's forward the port and test the service:

```bash
kubectl port-forward service/subscription 9696:80
```

In another terminal, run the test script again:

```bash
python q6_test.py
```

You should get the same result as in Question 1.

### Autoscaling

Let's create a HorizontalPodAutoscaler (HPA):

```bash
kubectl autoscale deployment subscription --name subscription-hpa --cpu-percent=20 --min=1 --max=3
# Check the status of the HPA
kubectl get hpa
```

If the HPA doesn't run properly, install the Metrics Server:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Increase the Load

Let's modify the test script to continuously send requests:

```python
import requests
import time

url = "http://localhost:9696/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

while True:
    time.sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)
```

Save this as load_test.py and run it:

```bash
python load_test.py
```

## Question 8: Maximum Replicas

Let's monitor the HPA to see how it reacts to the increased load:

```bash
kubectl get hpa subscription-hpa --watch
```

The maximum amount of replicas during this test will be the answer to Question 8.