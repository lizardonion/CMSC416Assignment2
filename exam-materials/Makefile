# CMSC416 A2 Makefile
AN = a2
CLASS = 416
GS_COURSE_ID = 1239306
GS_ASSIGN_ID = 7678696

COPTS = -g -Wall -O3 -fstack-protector-all -Wno-unused-variable -Wno-unused-result
GCC = gcc $(COPTS)
MPICC = mpicc $(COPTS)
SHELL  = /bin/bash
.SHELLFLAGS = -O nullglob -c
CWD    = $(shell pwd | sed 's/.*\///g')

PROGRAMS = \
	mpi_hello \
	heat_serial \
	heat_mpi    \
	kmeans_serial \
	kmeans_mpi \

all : $(PROGRAMS)

# cleaning target to remove compiled programs/objects
clean :
	rm -f $(PROGRAMS) *.o vgcore.*

export PARALLEL?=False		# disable parallel testing if not overridden

help :
	@echo 'Typical usage is:'
	@echo '  > make                          # build all programs'
	@echo '  > make clean                    # remove all compiled items'
	@echo '  > make clean-tests              # remove all temporary testing files'
	@echo '  > make zip                      # create a zip file for submission'
	@echo '  > make prob1                    # built targets associated with problem 1'
	@echo '  > make test-setup               # prepare for testing, download data files'
	@echo '  > make test                     # run all tests'
	@echo '  > make test-prob2               # run test for problem 2'
	@echo '  > make test-prob2 testnum=5     # run problem 2 test #5 only'
	@echo '  > make update                   # download and install any updates to project files'
	@echo '  > make submit                   # upload submission to Gradescope'


############################################################
# 'make zip' to create complete.zip for submission
ZIPNAME = $(AN)-complete.zip
zip : clean clean-tests
	rm -f $(ZIPNAME)
	cd .. && zip "$(CWD)/$(ZIPNAME)" -r "$(CWD)" --exclude \*/sample-mnist-data/\*
	@echo Zip created in $(ZIPNAME)
	@if (( $$(stat -c '%s' $(ZIPNAME)) > 10*(2**20) )); then echo "WARNING: $(ZIPNAME) seems REALLY big, check there are no abnormally large test files"; du -h $(ZIPNAME); fi
	@if (( $$(unzip -t $(ZIPNAME) | wc -l) > 256 )); then echo "WARNING: $(ZIPNAME) has 256 or more files in it which may cause submission problems"; fi
	@echo

############################################################
# `make update` to get project updates
update :
ifeq ($(findstring solution,$(CWD)),)
	curl -s https://www.cs.umd.edu/~profk/$(CLASS)/$(AN)-update.sh | /bin/bash 
else
	@echo "Cowardly refusal to update solution"
endif

################################################################################
# `make submit` to upload to gradescope
submit : zip 
	@chmod u+x gradescope-submit
	@echo '=== SUBMITTING TO GRADESCOPE ==='
	./gradescope-submit $(GS_COURSE_ID) $(GS_ASSIGN_ID) $(ZIPNAME)

################################################################################
# testing targets
test : test-prob1 test-prob2 test-prob3


# data files that are downloaded for testing; large so do not include
# in distribution or submission
DATA_NAME=sample-mnist-data
DATA_FILE="$(DATA_NAME).zip"
DATA_URL="https://www.cs.umd.edu/~profk/416/$(DATA_FILE)"

test-setup:
	@chmod u+x testy
	@if [[ ! -x $(DATA_NAME) ]]; then \
	  wget $(DATA_URL); \
	  unzip $(DATA_FILE); \
	  rm $(DATA_FILE);    \
	fi

clean-tests :
	rm -rf test-results

################################################################################
# Demo MPI program
mpi_hello : mpi_hello.c
	$(MPICC) -o $@ $^

################################################################################
# Problem 1 : heat
prob1: heat_serial heat_mpi

heat_serial : heat_serial.c
	$(GCC) -o $@ $^

heat_mpi : heat_mpi.c
	$(MPICC) -o $@ $^

test-prob1: test-setup heat_mpi
	./testy test_heat.org $(testnum)

################################################################################
# Problem 2 : kmeans_serial

prob2: kmeans_serial

kmeans_serial : kmeans_serial.c kmeans_util.c
	$(GCC) -o $@ $^ -lm

test-prob2: test-setup prob2 
	./testy test_kmeans_serial.org $(testnum)

################################################################################
# Problem 3 : kmeans_mpi
prob3: kmeans_mpi

kmeans_mpi : kmeans_mpi.c kmeans_util.c
	$(MPICC) -o $@ $^ -lm

test-prob3: test-setup prob3 test_kmeans_mpi.org
	./testy test_kmeans_mpi.org $(testnum)


