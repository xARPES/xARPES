.PHONY: all clean

all: xarpes.png xarpes_black.png

xarpes.png: xarpes.svg
	inkscape -w 992 -o $@ $<

xarpes_black.png: xarpes.svg
	inkscape -w 992 -b black -o $@ $<

clean:
	rm -f *.png
