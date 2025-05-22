# This helps inspect playwright mcp server inside the docker container
# You may open a terminal in the container and run the following command to inspect the playwright server
# npx -y @modelcontextprotocol/inspector npx @playwright/mcp@latest --no-sandbox --config /deps/langgraph-mcp/playwright/config.json
# due to the 6274 and 6277 port bindings, you'll be able to inspect the server at http://localhost:6274
docker run \
  --hostname=639bfdb152e9 \
  --env=TEST_CLAIM_ID=375491184 \
  --env="POSTGRES_URI=postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable" \
  --env=OPENAI_API_KEY=${OPENAI_API_KEY} \
  --env=LANGSMITH_ENDPOINT=https://api.smith.langchain.com \
  --env=LANGSMITH_TRACING=true \
  --env=MCP_SERVER_CONFIG=./.mcp-servers-config.json \
  --env=LANGSMITH_PROJECT=langgraph-mcp \
  --env=REDIS_URI=redis://langgraph-redis:6379 \
  --env=SMITHERY_API_KEY=${SMITHERY_API_KEY} \
  --env=LANGSMITH_API_KEY=${LANGSMITH_API_KEY} \
  --env=PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --env=LANG=C.UTF-8 \
  --env=GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305 \
  --env=PYTHON_VERSION=3.12.10 \
  --env=PYTHON_SHA256=07ab697474595e06f06647417d3c7fa97ded07afc1a7e4454c5639919b46eaea \
  --env=PYTHONFAULTHANDLER=1 \
  --env=PYTHONUNBUFFERED=True \
  --env=PORT=8000 \
  --env=PIP_ROOT_USER_ACTION=ignore \
  --env=N_WORKERS=1 \
  --env=N_JOBS_PER_WORKER=10 \
  --env=LANGSMITH_LANGGRAPH_API_REVISION=4559315 \
  --env=LANGSMITH_LANGGRAPH_API_VARIANT=licensed \
  --env=LANGGRAPH_RUNTIME_EDITION=postgres \
  --env='LANGSERVE_GRAPHS={"assist_with_planner": "/deps/langgraph-mcp/src/langgraph_mcp/with_planner/graph.py:graph"}' \
  --network=langgraph-mcp_default \
  --workdir=/deps/langgraph-mcp \
  -p 8123:8000 \
  -p 6274:6274 \
  -p 6277:6277 \
  --restart=no \
  --label='com.docker.compose.config-hash=32caae03a0c91ffc11e1b2ce02f245cc2befc78178d72eb003997f3a5bbedcb0' \
  --label='com.docker.compose.container-number=1' \
  --label='com.docker.compose.depends_on=langgraph-postgres:service_healthy:false,langgraph-redis:service_healthy:false' \
  --label='com.docker.compose.image=sha256:71f2cbd4b73430de0317d5f7b9f7e36c76f9aff5bf6200429979b512f0b03f5b' \
  --label='com.docker.compose.oneoff=False' \
  --label='com.docker.compose.project=langgraph-mcp' \
  --label='com.docker.compose.project.config_files=/Users/pdhoolia/gh/pdhoolia/langgraph-mcp/docker-compose.yml,-' \
  --label='com.docker.compose.project.working_dir=/Users/pdhoolia/gh/pdhoolia/langgraph-mcp' \
  --label='com.docker.compose.replace=dbcbd319184b5a297b54e9bea81e7896dfc4daecfac239d98473ba8bd9bc9915' \
  --label='com.docker.compose.service=langgraph-api' \
  --label='com.docker.compose.version=2.36.0' \
  --label='org.opencontainers.image.revision=4559315' \
  --runtime=runc \
  -d \
  langgraph-mcp-langgraph-api