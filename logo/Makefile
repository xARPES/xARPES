.PHONY: all clean

all: xarpes.png xarpes_small.png xarpes_black.png xarpes_white.png

xarpes.png: xarpes.svg
	inkscape -w 948 -o $@ $<

xarpes_small.svg: xarpes.svg
	cp $< $@
	sed -i 's/23.7/7.7/' $@
	sed -i 's/474/154/' $@
	sed -i '/M\(7\|11\|15\|19\)/d' $@

xarpes_small.png: xarpes_small.svg
	inkscape -w 308 -o $@ $<

xarpes_black.png: xarpes.svg
	inkscape -w 948 -b black -o $@ $<

xarpes_white.png: xarpes.svg
	inkscape -w 948 -b white -o $@ $<

clean:
	rm -f *.png *_small.svg
