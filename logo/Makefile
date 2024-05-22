.PHONY: all clean

all: xarpes.png xarpes_black.png xarpes_white.png xarpes_small.png xarpes_small_black.png xarpes_small_white.png xarpes_square.png

xarpes_small.svg: xarpes.svg
	cp $< $@
	sed -i 's/23.5/7.5/' $@
	sed -i 's/470/150/' $@
	sed -i '/M\(7\|11\|15\|19\)/d' $@

xarpes.png: xarpes.svg
	inkscape -w 940 -o $@ $<

xarpes_black.png: xarpes.svg
	inkscape -w 940 -b black -o $@ $<

xarpes_white.png: xarpes.svg
	inkscape -w 940 -b white -o $@ $<

xarpes_small.png: xarpes_small.svg
	inkscape -w 600 -o $@ $<

xarpes_small_black.png: xarpes_small.svg
	inkscape -w 600 -b black -o $@ $<

xarpes_small_white.png: xarpes_small.svg
	inkscape -w 600 -b white -o $@ $<

xarpes_square.png: xarpes_small_white.png
	convert $< -gravity center -extent 800x800 $@

clean:
	rm -f *.png *_small.svg
