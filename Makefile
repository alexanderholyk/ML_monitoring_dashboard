# Makefile

# ------------------------
# Names
# ------------------------
NETWORK := sentinet
VOLUME  := sentiment_logs

API_IMAGE := sentiment-api
DASH_IMAGE := sentiment-monitor

API_CT := sentiment_api
DASH_CT := sentiment_monitor

API_PORT := 8000
DASH_PORT := 8501

# ------------------------
# Helpers
# ------------------------
.PHONY: build run clean ps logs logs-once eval wait api-up

# Simple check that Docker is running
DOCKER_OK := $(shell docker info >/dev/null 2>&1 && echo 1 || echo 0)

build:
ifeq ($(DOCKER_OK),0)
	@echo "Docker is not running. Start Docker Desktop and retry."; exit 1
endif
	# Build API image
	docker build -f api/Dockerfile -t $(API_IMAGE) .
	# Build Monitoring image
	docker build -f monitoring/Dockerfile -t $(DASH_IMAGE) .

run: build
	-@docker network create $(NETWORK) >/dev/null 2>&1 || true
	-@docker volume create $(VOLUME) >/dev/null 2>&1 || true
	-@docker rm -f $(API_CT) >/dev/null 2>&1 || true
	-@docker rm -f $(DASH_CT) >/dev/null 2>&1 || true
	# Start API
	docker run -d --name $(API_CT) --network $(NETWORK) \
		-v $(VOLUME):/app/logs \
		-p $(API_PORT):8000 \
		$(API_IMAGE)
	# Wait for API to be up (poll /docs on localhost)
	$(MAKE) api-up
	# Start Streamlit dashboard
	docker run -d --name $(DASH_CT) --network $(NETWORK) \
		-v $(VOLUME):/app/logs \
		-p $(DASH_PORT):8501 \
		$(DASH_IMAGE)
	# Run evaluation INSIDE the API container so it can hit 127.0.0.1:8000 and write logs to the shared volume
	docker exec -it $(API_CT) sh -c 'python /app/evaluate.py --api http://127.0.0.1:8000/predict --test /app/test_data.json'
	@echo ""
	@echo "API       -> http://localhost:$(API_PORT)/docs"
	@echo "Dashboard -> http://localhost:$(DASH_PORT)"

# Wait until API answers on localhost (port-mapped)
api-up:
	@echo "Waiting for API to become ready on http://localhost:$(API_PORT)/docs ..."
	@bash -c 'for i in $$(seq 1 60); do \
		if curl -fsS http://localhost:$(API_PORT)/docs >/dev/null; then \
			echo "API is up."; exit 0; \
		fi; \
		sleep 1; \
	done; echo "API did not become ready in time." >&2; exit 1'

ps:
	docker ps --filter "name=$(API_CT)" --filter "name=$(DASH_CT)"

logs:
	@echo "Tailing API logs (Ctrl-C to stop) ..."
	docker logs -f $(API_CT)

logs-once:
	docker logs --tail=200 $(API_CT)

eval:
	# Re-run the evaluator to append more logs and refresh dashboard
	docker exec -it $(API_CT) sh -c 'python /app/evaluate.py --api http://127.0.0.1:8000/predict --test /app/test_data.json'

clean:
	-@docker rm -f $(API_CT) >/dev/null 2>&1 || true
	-@docker rm -f $(DASH_CT) >/dev/null 2>&1 || true
	-@docker network rm $(NETWORK) >/dev/null 2>&1 || true
	-@docker volume rm $(VOLUME) >/dev/null 2>&1 || true
	@echo "Cleaned containers, network, and volume."