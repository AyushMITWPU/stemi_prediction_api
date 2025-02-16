{pkgs}: {
  deps = [
    pkgs.openssh
    pkgs.hdf5
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.glibcLocales
  ];
}
