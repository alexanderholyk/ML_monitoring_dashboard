# Makefile


### Variables

# Shared Docker network
NETWORK := sentinet
# Shared log volume
VOLUME := sentiment_logs
# API Docker image
API_IMAGE := sentiment-api
# Dashboard Docker image
DASH_IMAGE := sentiment-monitor
# API container name
API_CT := sentiment_api
# Dashboard container name
DASH_CT := sentiment_monitor
# API port (localhost)
API_PORT := 8000
# Dashboard port (localhost)
DASH_PORT := 8501


### Helpers
.PHONY: build run clean

# Simple check that Docker is running
DOCKER_OK := $(shell docker info >/dev/null 2>&1 && echo 1 || echo 0)

# Build Docker images for API and Dashboard
build:
ifeq ($(DOCKER_OK),0)
	@echo "Docker is not running. Start Docker Desktop and retry."; exit 1
endif
	# Build API image
	docker build -f api/Dockerfile -t $(API_IMAGE) .
	# Build Monitoring image
	docker build -f monitoring/Dockerfile -t $(DASH_IMAGE) .

# Run API + Dashboard + initial evaluation
# treats build as a prerequisite; if build hasn't been run, it will be called first
run: build
	@echo "Starting services..."
	# Create network & volume if missing
	-@docker network create $(NETWORK) >/dev/null 2>&1 || true
	-@docker volume create $(VOLUME)  >/dev/null 2>&1 || true
	# Remove old containers
	-@docker rm -f sentiment_api sentiment_monitor >/dev/null 2>&1 || true
	# Start API
	docker run -d --name sentiment_api --network $(NETWORK) -v $(VOLUME):/app/logs -p $(API_PORT):8000 $(API_IMAGE)
	# Wait for API to be ready
	sleep 5
	# Start Dashboard
	docker run -d --name sentiment_monitor --network $(NETWORK) -v $(VOLUME):/app/logs -p $(DASH_PORT):8501 $(DASH_IMAGE)
	# Run initial evaluation to populate logs
	docker exec $(API_CT) python evaluate.py --api http://127.0.0.1:8000/predict --test test_data.json
	@echo ""
	@echo "API        -> http://localhost:$(API_PORT)/docs"
	@echo "Dashboard  -> http://localhost:$(DASH_PORT)"

# Remove all containers, network, and volume
clean:
	-@docker rm -f $(API_CT) $(DASH_CT) >/dev/null 2>&1 || true
	-@docker network rm $(NETWORK) >/dev/null 2>&1 || true
	-@docker volume rm $(VOLUME) >/dev/null 2>&1 || true
	@echo "Cleaned containers, network, and volume."
