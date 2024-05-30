.PHONY: clean html

html:
	$(MAKE) -C ../logo/ xarpes_small.svg
	sphinx-build -M html . .
	mkdir -p html/_images/
	cp ../logo/xarpes.svg html/_images/
	touch html/.nojekyll

clean:
	rm -rf doctrees html
