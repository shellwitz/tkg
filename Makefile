.PHONY: test
test:
	.venv/bin/python -m unittest discover -s tests

.PHONY: web_show
web_show:
	.venv/bin/streamlit run scripts/streamlit_chunk_retrieval.py --server.port 8501 --server.headless true

define INHIBIT
	@if command -v systemd-inhibit >/dev/null 2>&1; then \
		systemd-inhibit --why="$(1)" --mode=block bash -lc '$(2)'; \
	else \
		echo "No systemd-inhibit available. Figure out how to run the scripts over a longer period of time without your pc going to sleep/hibernate yourself."; \
		bash -lc '$(2)'; \
	fi
endef

.PHONY: overnight-ingest
overnight-ingest:
	$(call INHIBIT,overnight tk rag insert,source .venv/bin/activate && python scripts/ingest_test.py -fb -q)

.PHONY: overnight-answer
overnight-answer:
	$(call INHIBIT,overnight tk rag answering,source .venv/bin/activate && python scripts/answer_test.py -qs)