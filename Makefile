.PHONY: install data run dashboard test cov eval clean
install:
	pip install -r requirements-dev.txt
data:          ## 3,000 simulated transactions, Bank and ERP written as Excel
	python generate_data.py --num-transactions 3000 --excel BANK,ERP
run:
	python run_pipeline.py
dashboard:
	streamlit run dashboard.py
test:
	pytest
cov:
	pytest --cov=recon --cov=dashboard --cov-report=term-missing
eval:          ## grade the pipeline against the simulator's held-out labels
	python validate_ground_truth.py
clean:
	rm -rf .pytest_cache .coverage htmlcov data/raw/* data/reconciliation.db
