{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.allowUnsupportedSystem = true;
          config.cudaSupport = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            ffmpeg
            cudaPackages.cudatoolkit
            nvidia-docker
            uv
          ];
          buildInputs = with pkgs; [
          ];

          LD_LIBRARY_PATH = "${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH";
          shellHook = ''
            export XLA_FLAGS=--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}
          '';
        };
      }
    );
}

