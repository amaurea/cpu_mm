SHELL=/bin/bash
.PHONY: build inline clean
inline: build
	(shopt -s nullglob; cd python; rm -f *.so; ln -s ../build/*.so ../build/*.dylib .)
build: build/build.ninja
	(cd build; meson compile -v)
build/build.ninja: Makefile meson.build
	rm -rf build
	mkdir build
	meson setup build --buildtype=debug
clean:
	rm -rf build
