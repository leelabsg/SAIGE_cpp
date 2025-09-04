FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    # C++ compiler and tools
    g++ \
    gcc \
    make \
    # Required libraries
    libeigen3-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    libarmadillo-dev \
    # R dependencies (for RcppArmadillo)
    r-base-dev \
    r-cran-rcpp \
    r-cran-rcpparmadillo \
    r-cran-rcppparallel \
    # Additional utilities
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libeigen3-dev \
    libsuperlu-dev \
    && rm -rf /var/lib/apt/lists/*
 
# Install yaml-cpp from source
RUN cd /tmp && \
    wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.tar.gz && \
    tar -xzf yaml-cpp-0.7.0.tar.gz && \
    cd yaml-cpp-yaml-cpp-0.7.0 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_BUILD_SHARED_LIBS=ON .. && \    
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/yaml-cpp*

# Install cxxopts (header-only library)
RUN cd /tmp && \
    wget https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz && \
    tar -xzf v3.1.1.tar.gz && \
    cd cxxopts-3.1.1 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make install && \
    cd / && rm -rf /tmp/cxxopts*

# Set environment variables for compilation
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=/usr/local/include:$CPLUS_INCLUDE_PATH

# Set OpenMP environment
ENV OMP_NUM_THREADS=4

# Create working directory
WORKDIR /app

# Copy source files
COPY *.cpp *.hpp ./

# Create a simple Makefile for compilation
RUN echo 'CXX = g++' > Makefile && \
    echo 'CXXFLAGS = -std=c++17 -O3 -fopenmp -DARMA_64BIT_WORD=1 -DRCPP_PARALLEL_USE_TBB=0' >> Makefile && \
    echo 'INCLUDES = -I/usr/local/include -I/usr/share/R/include -I/usr/include/eigen3 -I$(shell Rscript -e "cat(system.file(\"include\", package=\"Rcpp\"))") -I$(shell Rscript -e "cat(system.file(\"include\", package=\"RcppArmadillo\"))") -I$(shell Rscript -e "cat(system.file(\"include\", package=\"RcppParallel\"))")' >> Makefile && \
    echo 'LIBS = -lyaml-cpp -larmadillo -lopenblas -llapack -lsuperlu -ltbb -fopenmp -L$(shell Rscript -e "cat(R.home())")/lib -lR' >> Makefile && \
    echo 'SOURCES = $(wildcard *.cpp)' >> Makefile && \
    echo 'TARGET = saige-null' >> Makefile && \
    echo '' >> Makefile && \
    echo '$(TARGET): $(SOURCES)' >> Makefile && \
    echo '\t$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)' >> Makefile && \
    echo '' >> Makefile && \
    echo 'clean:' >> Makefile && \
    echo '\trm -f $(TARGET)' >> Makefile && \
    echo '' >> Makefile && \
    echo '.PHONY: clean' >> Makefile

# Build the application
RUN make

# Set the default command
CMD ["./saige-null", "--help"]