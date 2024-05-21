.PHONY: all clean

all: xarpes.png xarpes_black.png xarpes_white.png xarpes_small.png xarpes_small_black.png xarpes_small_white.png

xarpes_small.svg: xarpes.svg
	cp $< $@
	sed -i 's/23.4/7.4/' $@
	sed -i 's/468/148/' $@
	sed -i '/M\(7\|11\|15\|19\)/d' $@

xarpes.png: xarpes.svg
	inkscape -w 936 -o $@ $<

xarpes_black.png: xarpes.svg
	inkscape -w 936 -b black -o $@ $<

xarpes_white.png: xarpes.svg
	inkscape -w 936 -b white -o $@ $<

xarpes_small.png: xarpes_small.svg
	inkscape -w 592 -o $@ $<

xarpes_small_black.png: xarpes_small.svg
	inkscape -w 592 -b black -o $@ $<

xarpes_small_white.png: xarpes_small.svg
	inkscape -w 592 -b white -o $@ $<

clean:
	rm -f *.png *_small.svg
