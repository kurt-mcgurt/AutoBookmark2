{ pkgs }: {
  deps = [
    pkgs.python312Full
    pkgs.ghostscript
    pkgs.pdftk
    pkgs.stdenv.cc.cc.lib
  ];
}