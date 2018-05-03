
.PHONY: test

test:
	PYTHONPATH=./ nosetests --nologcapture --with-process-isolation -v -s test/*
