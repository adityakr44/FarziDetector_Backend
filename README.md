# FakerBackend

## Build a Docker Image

Make sure you have docker installed on your system then run the following command
Also change the host ip to 0.0.0.0 and port to 5000

`docker build --tag <image_tag> <path_to_folder>`

## Create a container and run the image

To create container from the image run the following command

`docker run -p 5000:5000 -d <image_tag>`

## Test the container

Open your browser and open the link `http://localhost:5000/` to see results

## Make predictions

To make preditctions on the image send a post request with following parameters to the request url `http://localhost:5000/predict`

`headers: {'Accept': 'application/json', 'Content-Type': 'multipart/form-data'}`
`data: {'name': <image_name>, 'type': 'image/<type>', 'uri': <image_uri>}`
