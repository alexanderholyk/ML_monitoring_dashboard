# Makefile for Sentiment Monitoring

# Variables
API_IMAGE := sentiment-api           # Docker image name for API service
DASH_IMAGE := sentiment-monitor      # Docker image name for dashboard service
NETWORK := sentinet                  # Shared Docker network name
VOLUME := sentiment_logs             # Shared Docker volume for logs
API_PORT := 8000                     # Localhost port for API
DASH_PORT := 8501                    # Localhost port for Streamlit dashboard

# Build Docker images for API and Dashboard
build:
	@echo "Building API and Dashboard images..."
	docker build -f api/Dockerfile -t $(API_IMAGE) .
	docker build -f monitoring/Dockerfile -t $(DASH_IMAGE) .

# Run services and evaluation:
# Creates a Docker network and volume if they don't exist,
# starts the API and dashboard containers, and runs the evaluation script.
# The evaluation script sends a request to the API with test data and logs the results.
# It also prints the URLs for the API and dashboard.
# The API is expected to be running at http://localhost:8000
run: build
	@echo "Starting services..."
	-@docker network create $(NETWORK) >/dev/null 2>&1 || true
	-@docker volume create $(VOLUME) >/dev/null 2>&1 || true
	-@docker rm -f sentiment_api sentiment_monitor >/dev/null 2>&1 || true
	docker run -d --name sentiment_api --network $(NETWORK) \
		-v $(VOLUME):/app/logs -p $(API_PORT):8000 $(API_IMAGE)
	sleep 5
	docker run -d --name sentiment_monitor --network $(NETWORK) \
		-v $(VOLUME):/app/logs -p $(DASH_PORT):8501 $(DASH_IMAGE)
	docker exec sentiment_api python /app/evaluate.py \
		--api http://127.0.0.1:8000/predict \
		--test /app/test_data.json
	@echo ""
	@echo "API running at: http://localhost:$(API_PORT)/docs"
	@echo "Dashboard running at: http://localhost:$(DASH_PORT)"

# Remove containers, images, network, and volume
clean:
	@echo "Cleaning up containers, images, network, and volume..."
	-@docker rm -f sentiment_api sentiment_monitor >/dev/null 2>&1 || true
	-@docker rmi $(API_IMAGE) $(DASH_IMAGE) >/dev/null 2>&1 || true
	-@docker network rm $(NETWORK) >/dev/null 2>&1 || true
	-@docker volume rm $(VOLUME) >/dev/null 2>&1 || true