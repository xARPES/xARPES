.PHONY: all clean

all: xarpes.png xarpes_black.png xarpes_white.png xarpes_small.png xarpes_small_black.png xarpes_small_white.png xarpes_square.png exubi.png exubi_black.png exubi_white.png

xarpes_small.svg: xarpes.svg
	cp $< $@
	sed -i 's/23.5/7.5/' $@
	sed -i 's/470/150/' $@
	sed -i '/M\(7\|11\|15\|19\)/d' $@

%.png: %.svg
	inkscape -w 940 -o $@ $<

%_black.png: %.svg
	inkscape -w 940 -b black -o $@ $<

%_white.png: %.svg
	inkscape -w 940 -b white -o $@ $<

%_small.png: %_small.svg
	inkscape -w 600 -o $@ $<

%_small_black.png: %_small.svg
	inkscape -w 600 -b black -o $@ $<

%_small_white.png: %_small.svg
	inkscape -w 600 -b white -o $@ $<

%_square.png: %_small_white.png
	convert $< -gravity center -extent 700x700 $@

clean:
	rm -f *.png *_small.svg
