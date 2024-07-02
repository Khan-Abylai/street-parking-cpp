base_image=registry.infra.smartparking.kz/parking_cpp_orangepi_base:1.1
app_image=registry.infra.smartparking.kz/street-parking
tag=v2.0.0


prod:
	rm -f ./prod && mkdir ./prod && cp -r src/ prod/ \
	&& cp -r models/ prod/ && cp -r Dockerfile prod/ \
	&& cp CMakeLists.txt prod/ && cp config.json prod/ \
	&& cp docker-compose.yaml prod/ && cp Makefile prod/ \
	&& echo "Production prepared"

build_app:
	docker build -t ${app_image}:${tag} -f Dockerfile .


