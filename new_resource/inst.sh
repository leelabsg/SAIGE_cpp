docker run --rm -it \
  saige-cpp:latest \
  ./saige-null -h

docker run --rm -it \
  -v /media/leelabsg-storage0/UKBB_WORK/SAIGE_cpp/SAIGE/extdata/:/data \
  saige-cpp:latest \
  ./saige-null -c /data/step1_input.yaml -d /data/input/pheno_1000samples.txt -o fit.nthreads=16


docker build -t saige-cpp  . 
docker run --rm -it   -v /media/leelabsg-storage0/UKBB_WORK/SAIGE_cpp/SAIGE/extdata/:/data   saige-cpp:latest   ./saige-null -c /data/step1_input.yaml -d /data/input/pheno_1000samples.txt -o fit.nthreads=16

step1_fitNULLGLMM.R   \
--plinkFile=./input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly \
--phenoFile=./input/pheno_1000samples.txt \
--phenoCol=y \
--covarColList=x1,x2  \
--sampleIDColinphenoFile=IID \
--traitType=binary \
--outputPrefix=./output/example_out \
--nThreads=4 \
--LOCO=FALSE  \
--minMAFforGRM=0.01   \
--skipModelFitting=FALSE  \
--tol=0.01 \
--isCovariateOffset=FALSE \
--skipVarianceRatioEstimation=TRUE  