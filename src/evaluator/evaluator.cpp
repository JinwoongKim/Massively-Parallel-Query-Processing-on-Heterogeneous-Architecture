#include "evaluator/evaluator.h"

#include "common/macro.h"
#include "common/logger.h"
#include "tree/mphr.h"
#include "tree/hybrid.h"
#include "tree/rtree.h"

#include <cassert>
#include <unistd.h>
#include <locale> 
#include <thread> 

namespace ursus {
namespace evaluator {

/**
 * @brief Return the singleton evaluator instance
 */
Evaluator& Evaluator::GetInstance(){
  static Evaluator evaluator;
  return evaluator;
}

bool Evaluator::Initialize(int argc, char** argv){
  bool ret;

  // Parse the args and initialize variables
  if( !ParseArgs(argc, argv)) {
    PrintHelp(argv);
    return false;
  }

  // Read dataset based on initialized variables
  ret=ReadDataSet();
  assert(ret);

  if(number_of_search > 0) {
    ret=ReadQuerySet();
    assert(ret);
  }

  return true;
}

bool Evaluator::ReadDataSet(void){
  auto low_data_type = ToLowerCase(data_type);
  if( low_data_type == "real"){
    input_data_set.reset(new io::DataSet(GetNumberOfDims(), number_of_data,
          "/home/jwkim/dataFiles/input/real/NOAA0.bin",
          DATASET_TYPE_BINARY, DATA_TYPE_REAL)); 
  } else if (low_data_type == "synthetic") {
    input_data_set.reset(new io::DataSet(GetNumberOfDims(), number_of_data,
          "/home/jwkim/dataFiles/input/synthetic/synthetic_200m_3d_data.bin",
          DATASET_TYPE_BINARY, DATA_TYPE_SYNTHETIC)); 
  } else {
    assert(0);
  }
  return true;
}

bool Evaluator::ReadQuerySet(void){
  //TODO : hard coded now...
  auto low_data_type = ToLowerCase(data_type);
  if( low_data_type == "real"){
    query_data_set.reset(new io::DataSet(GetNumberOfDims(), number_of_search*2,
          "/home/jwkim/dataFiles/query/real/real_dim_query.3.bin."
          +selectivity+"s."+query_size, DATASET_TYPE_BINARY, DATA_TYPE_REAL)); 
  } else if (low_data_type == "synthetic") {
    query_data_set.reset(new io::DataSet(GetNumberOfDims(), number_of_search*2,
          "/home/jwkim/dataFiles/query/synthetic/synthetic_dim_query.3.bin."
          +selectivity+"s", DATASET_TYPE_BINARY, DATA_TYPE_SYNTHETIC)); 
  } else {
    assert(0);
  }

  return true;
}


int Evaluator::SetDevice() {

  int number_of_gpus;
  cudaGetDeviceCount(&number_of_gpus);

  for(ui range(gpu_itr, 0, number_of_gpus)) {
    cudaDeviceProp prop;

    // NOTE :: There are many other fields in the cudaDeviceProp struct which
    // describe the amounts of various types of memory, limits on thread block
    // sizes, and many other characteristics of the GPU. 
    cudaGetDeviceProperties(&prop, gpu_itr);

    // attempt to set the device
    cudaError_t error = cudaSetDevice(gpu_itr);
     // fail
    if( error != cudaSuccess) {
      continue;
    }

    size_t avail, total;
    cudaMemGetInfo( &avail, &total );
    // if available space is less than 10%, try to get another on,
    // otherwise success to get the GPU
    if( avail > 0.1 ) {
      if((gpu_itr+1)%10==1) {
        LOG_INFO("%ust GPU(%s) is selected", gpu_itr+1, prop.name);
      } else if((gpu_itr+1)%10==2) {
        LOG_INFO("%und GPU(%s) is selected", gpu_itr+1, prop.name);
      } else if((gpu_itr+1)%10==3) {
        LOG_INFO("%urd GPU(%s) is selected", gpu_itr+1, prop.name);
      } else {
        LOG_INFO("%uth GPU(%s) is selected", gpu_itr+1, prop.name);
      }
      return gpu_itr;
    }
  }

  LOG_INFO("Unfortunately, none of devices is available in this machine");
  return -1;
}

/**
 * @brief Build all indexing structures in tree_queue with input_dataset
 * @return true if building all indexing structures successfully,
 *  otherwise return false 
 */
bool Evaluator::Build(void) {
  for(auto& tree : trees) {
    switch(tree->GetTreeType()){
      case TREE_TYPE_HYBRID:  {
        // Casting type from base class to derived class using dynamic_pointer_cast since it's shared_ptr
        std::shared_ptr<tree::Hybrid> hybrid = std::dynamic_pointer_cast<tree::Hybrid>(tree);
        hybrid->SetScanType(scan_type);
        hybrid->SetChunkSize(chunk_size);
        hybrid->SetNumberOfCUDABlocks(number_of_cuda_blocks);
        hybrid->SetNumberOfCPUThreads(number_of_cpu_threads);
        tree->Build(input_data_set);
        } break;
      case  TREE_TYPE_MPHR: {
        std::shared_ptr<tree::MPHR> mphr = std::dynamic_pointer_cast<tree::MPHR>(tree);
        mphr->SetNumberOfCUDABlocks(number_of_cuda_blocks);
        mphr->SetNumberOfPartition(number_of_partition);
        tree->Build(input_data_set);
        } break;
      case  TREE_TYPE_RTREE: {
        std::shared_ptr<tree::Rtree> rtree = std::dynamic_pointer_cast<tree::Rtree>(tree);
        rtree->SetNumberOfCPUThreads(number_of_cpu_threads);
        tree->Build(input_data_set);
        } break;
      default:
        assert(0);
        break;
    }
  }
  return true;
}

/**
 * @brief explore indexing structure with given query file
 * @return true for now 
 */
bool Evaluator::Search(void) {
  if( number_of_search == 0 ) return false;

  std::vector<ui> cpu_thread_vec = {1,2,4,8,16,32};
  std::vector<ui> chunk_size_vec = {1, 2, 4, 8, 16, 32, 64, 128, 256,
                                    512, 768, 1024};
  std::vector<ui> cuda_block_vec = {1, 2, 4, 8, 16, 32, 64, 128};

  for(auto& tree : trees) {
    switch(tree->GetTreeType()) {
      case TREE_TYPE_HYBRID: {
        if( EvaluationMode ) {
          std::shared_ptr<tree::Hybrid> hybrid = std::dynamic_pointer_cast<tree::Hybrid>(tree);

          if( EvaluationMode == 2 ||
              EvaluationMode == 3) {
            for(auto cpu_thread_itr : cpu_thread_vec) {
              for(auto chunk_size_itr : chunk_size_vec) {
                auto cuda_block_per_cpu = 128/cpu_thread_itr;
                if( chunk_size_itr >= cuda_block_per_cpu) {
                  hybrid->SetChunkSize(chunk_size_itr);
                  hybrid->SetNumberOfCPUThreads(cpu_thread_itr);
                  hybrid->SetNumberOfCUDABlocks(128);
                  LOG_INFO("Evaluation Mode On CPU Thread %u CUDA Block %u Chunk Size %u", cpu_thread_itr, cuda_block_per_cpu, chunk_size_itr);
                  tree->Search(query_data_set, number_of_search, number_of_repeat);
                }
              }
            }
          }

          if( EvaluationMode == 1 ||
              EvaluationMode == 3 ) {
            for(auto cuda_block_itr : cuda_block_vec) {
              for(auto chunk_size_itr : chunk_size_vec) {
                auto cpu_thread = 1;
                if( chunk_size_itr >= cuda_block_itr) {
                  hybrid->SetChunkSize(chunk_size_itr);
                  hybrid->SetNumberOfCPUThreads(cpu_thread);
                  hybrid->SetNumberOfCUDABlocks(cuda_block_itr);
                  LOG_INFO("Evaluation Mode On CPU Thread %u CUDA Block %u Chunk Size %u", 1, cuda_block_itr, chunk_size_itr);
                  tree->Search(query_data_set, number_of_search, number_of_repeat);
                }
              }
            }
          }
        } else {
          tree->Search(query_data_set, number_of_search, number_of_repeat);
        }
      }  break;
      case TREE_TYPE_MPHR:
      case TREE_TYPE_MPHR_PARTITION: {
        if( EvaluationMode ) {
          std::shared_ptr<tree::MPHR> mphr = std::dynamic_pointer_cast<tree::MPHR>(tree);

          for(auto cuda_block_itr : cuda_block_vec) {
            mphr->SetNumberOfCUDABlocks(cuda_block_itr);
            LOG_INFO("Evaluation Mode On CUDA Block %u", cuda_block_itr);
            tree->Search(query_data_set, number_of_search, number_of_repeat);
          }
        } else {
          LOG_INFO("");
          tree->Search(query_data_set, number_of_search, number_of_repeat);
        }
      } break;
      case TREE_TYPE_RTREE: {
        if( EvaluationMode ) {
          std::shared_ptr<tree::Rtree> rtree = std::dynamic_pointer_cast<tree::Rtree>(tree);

          for(auto cpu_thread_itr : cpu_thread_vec) {
            rtree->SetNumberOfCPUThreads(cpu_thread_itr);
            LOG_INFO("Evaluation Mode On CPU Thread %u", cpu_thread_itr);
            tree->Search(query_data_set, number_of_search, number_of_repeat);
          }
        } else {
          tree->Search(query_data_set, number_of_search, number_of_repeat);
        }
      } break;
    }
  }

  return true;
}

//TODO :: Need to fix?  scrub
void Evaluator::PrintHelp(char **argv) const {
  std::cerr << "Usage:\n" << *argv << std::endl << 
  " -d number of data\n" 
  " [ -q number of queries, default : 0]\n" 
  " [ -b number of CUDA blocks, default 128]\n" 
  " [ -p number of CPU threads, default number of CPU cores]\n" 
  " [ -c chunk size(for hybrid), default : " << GetNumberOfDegrees() << "(number of degrees)]\n"
  " [ -s selection ratio(%), default : 0.01 (%) ]\n"
  " [ -l scan type(1: leaf, 2: extend leaf, 3: combine), default : leaf]\n"
  " [ -i index type(should be last), default : Hybrid-tree]\n"
  " [ -r number of repeat of search]\n" 
  " [ -e evaluation mode ]\n" 
  "\n e.g: ./bin/cuda -d 1000000 -q 1000 -s 0.5 -c 4\n" 
  << std::endl;
}

void Evaluator::PrintMemoryUsageOftheGPU() {
  cudaDeviceSynchronize();
  size_t used = GetUsedMem();
  size_t total = GetTotalMem();
  LOG_INFO(" Used Memory %lu(MB) / GPU Capacity %lu(MB) ( %.2f % )", 
           used/1000000, total/1000000, ( (double)used/(double)total)*100);
}

size_t Evaluator::GetUsedMem(void) {
  size_t avail, total;
  cudaMemGetInfo( &avail, &total );
  size_t used = total-avail;
  return used;
}

size_t Evaluator::GetAvailMem(void) {
  size_t avail, total;
  cudaMemGetInfo( &avail, &total );
  return avail;
}

size_t Evaluator::GetTotalMem(void) {
  size_t avail, total;
  cudaMemGetInfo( &avail, &total );
  return total;
}

//TODO :: Boost.Program_options
// http://www.boost.org/doc/libs/1_59_0/doc/html/program_options.html
bool Evaluator::ParseArgs(int argc, char **argv)  {

  // TODO scrubbing
  static const char *options="c:C:i:I:d:D:q:Q:b:B:p:P:s:S:l:L:r:R:e:E:t:T:y:Y:";
  std::string number_of_data_str;
  int current_option;

 
  while ((current_option = getopt(argc, argv, options)) != -1) {
    switch (current_option) {
      case 'i':
      case 'I': AddTrees(std::string(optarg)); break;
      case 'c':
      case 'C': chunk_size = atoi(optarg); break;
      case 'd':
      case 'D': number_of_data_str = std::string(optarg); break;
      case 'q':
      case 'Q': number_of_search = atoi(optarg); break;
      case 'b':
      case 'B': number_of_cuda_blocks = atoi(optarg); break;
      case 'p':
      case 'P': number_of_cpu_threads = atoi(optarg); break;
      case 's':
      case 'S': selectivity = std::string(optarg);  break;
      case 'l':
      case 'L': scan_type = (ScanType)atoi(optarg);  break;
      case 'r':
      case 'R': number_of_repeat = atoi(optarg);  break;
      case 'e':
      case 'E': EvaluationMode = atoi(optarg);  break;
      case 't':
      case 'T': number_of_partition = atoi(optarg);  break;
      case 'y':
      case 'Y': data_type = std::string(optarg);  break;
     default: break;
    } // end of switch
  } // end of while

  // check # of cuda blocks
  assert(number_of_cuda_blocks <= GetNumberOfMAXBlocks());

  // try to get the gpu
  int ret = SetDevice();
  // if failed to set the device, terminate the program
  if(ret == -1){ exit(1); }

  // Set default tree as a hybrid
  if(trees.empty()){ 
    AddTrees("hybrid");
  }

  // set the number of data and query
  if(number_of_data_str.empty()){ return false; }
  size_t position = number_of_data_str.find("m");
  if( position == std::string::npos )  {
    number_of_data = std::stoul(number_of_data_str);
  } else { 
    number_of_data = std::stoul(number_of_data_str.erase(position,1));
    number_of_data = number_of_data*1000000;
  }

  if( number_of_data <= 1000000 ){
    query_size="1m";
  } else if( number_of_data >= 40000000) {
    query_size="40m";
  } else  {
    query_size = std::to_string(number_of_data/1000000)+std::string("m");
  } 

  number_of_cpu_core = std::thread::hardware_concurrency();

  // set the default batch size for hybrid indexing to number of cpu core
  if( !number_of_cpu_threads ) {
    number_of_cpu_threads = number_of_cpu_core;
  }

  std::cout << *this << std::endl;
  return true;
}

void Evaluator::AddTrees(std::string _index_type) {
  // Make it lower case
  auto index_type = ToLowerCase(_index_type);

  if( index_type == "hybrid" ||
      index_type == "h") {
    std::shared_ptr<tree::Tree> tree (new tree::Hybrid());
    trees.push_back(tree);
  } else if ( index_type == "mphr" ||
              index_type == "m") {
    std::shared_ptr<tree::Tree> tree (new tree::MPHR());
    trees.push_back(tree);
  } else if ( index_type == "rtree" ||
              index_type == "r") {
    std::shared_ptr<tree::Tree> tree (new tree::Rtree());
    trees.push_back(tree);
  }
}

std::string Evaluator::ToLowerCase(std::string str) {
 std::string lower_str;
 std::locale loc;
 for (std::string::size_type range(i, 0, str.length())) {
   lower_str.append(sizeof(char), std::tolower(str[i],loc));
 }
 return lower_str;
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const Evaluator &evaluator) {
  os << " Evaluator : " << std::endl
     << " number of data = " << evaluator.number_of_data << std::endl
     << " number of degrees = " << GetNumberOfDegrees() << std::endl
     << " number of thread blocks = " << evaluator.number_of_cuda_blocks << std::endl
     << " number of threads = " << GetNumberOfThreads() << std::endl
     << " number of searches = " << evaluator.number_of_search << std::endl
     << " number of cpu cores = " << evaluator.number_of_cpu_core << std::endl
     << " number of CPU threads = " << evaluator.number_of_cpu_threads << std::endl
     << " scan type = " << ScanTypeToString(evaluator.scan_type) << std::endl
     << " chunk size = " << evaluator.chunk_size << std::endl
     << " selectivity = " << evaluator.selectivity << std::endl
     << " query size = " << evaluator.query_size << std::endl;

  return os;
}

} // End of evaluator namespace
} // End of ursus namespace
