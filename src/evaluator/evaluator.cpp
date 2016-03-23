#include "evaluator/evaluator.h"

#include "common/macro.h"
#include "common/logger.h"
#include "tree/mphr.h"
#include "tree/hybrid.h"

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
  //TODO : hard coded now...
  input_data_set.reset(new io::DataSet(GetNumberOfDims(), number_of_data,
                       //"/home/jwkim/dataFiles/input/real/NOAA0.bin",
                       "/home/jwkim/dataFiles/input/synthetic/synthetic_200m_3d_data.bin",
                       DATASET_TYPE_BINARY, DATA_TYPE_SYNTHETIC)); 
  return true;
}

bool Evaluator::ReadQuerySet(void){
  //TODO : hard coded now...
  query_data_set.reset(new io::DataSet(GetNumberOfDims(), number_of_search*2,
                       "/home/jwkim/dataFiles/query/real/real_dim_query.3.bin."
                       +selectivity+"s."+query_size,
                       DATASET_TYPE_BINARY, DATA_TYPE_SYNTHETIC)); 

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
    if(!tree->Build(input_data_set)) {
      return false;
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

  for(auto& tree : trees) {
    if(!tree->Search(query_data_set, number_of_search)) {
      return false;
    }
  }

  return true;
}

//TODO :: Need to fix?  scrub
void Evaluator::PrintHelp(char **argv) const {
  std::cerr << "Usage:\n" << *argv << std::endl << 
  " -d number of data\n" 
  " [ -q number of queries, default : 0]\n" 
  " [ -i index type, default : Hybrid-tree]\n"
//  " [ -m search algorithm type, 1: MPES, 2: MPTS, 3: MPHR, 4: MPHR2\n \
  5: Short-Stack, 6: Parent-Link, 7: Skip-Pointer ]\n"
  " [ -p partitioned version, number of block ]\n" 
  " [ -s selection ratio(%), default : 1 (%) ]\n"
  " [ -g number of gpus, default : 1 ]\n" 
  "\n e.g: ./bin/cuda -d 1000000 -q 1000 -s 0.5 -c 4 -w 3\n" 
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
  static const char *options="i:I:d:D:q:Q:p:P:s:S:g:G:m:M:";
  std::string number_of_data_str;
  int current_option;

  while ((current_option = getopt(argc, argv, options)) != -1) {
    switch (current_option) {
      case 'i':
      case 'I': AddTrees(std::string(optarg)); break;
//      case 'm':
//      case 'M': METHOD[atoi(optarg)-1] = true;
//                optind--;
//                for( ;optind < argc && *argv[optind] != '-'; optind++) {
//                  METHOD[atoi( argv[optind] )-1] = true;
//                }
//                break;
      case 'd':
      case 'D': number_of_data_str = std::string(optarg); break;
      case 'q':
      case 'Q': number_of_search = atoi(optarg); break;
      case 'p':
      case 'P': number_of_partitioned_tree = atoi(optarg); break;
      case 's':
      case 'S': selectivity = std::string(optarg);  break;
//      case 'g':
//      case 'G': number_of_gpus = atoi(optarg); break;
     default: break;
    } // end of switch
  } // end of while

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

//  if( METHOD[7] == true)
//    METHOD[0] = METHOD[1] = METHOD[2] =  METHOD[3] = METHOD[4] = METHOD[5] = METHOD[6] = true;
//
//
//  if( (METHOD[0] || METHOD[1] || METHOD[2] ||  METHOD[3] || METHOD[4] || METHOD[5] || METHOD[6]) && number_of_search == 0)
//  {
//    number_of_search = 1000;
//  }

  std::cout << *this << std::endl;
  return true;
}

void Evaluator::AddTrees(std::string _index_type) {
  // Make it lower case
  auto index_type = ToLowerCase(_index_type);

  if( index_type == "hybrid" ||
      index_type == "h") {
    auto tree = std::unique_ptr<tree::Tree>(new tree::Hybrid());
    trees.push_back(std::move(tree));
  } else if ( index_type == "mphr" ||
              index_type == "m") {
    auto tree = std::unique_ptr<tree::Tree>(new tree::MPHR());
    trees.push_back(std::move(tree));
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
     << " number of thread blocks = " << GetNumberOfBlocks() << std::endl
     << " number of threads = " << GetNumberOfThreads() << std::endl
     << " number of searches = " << evaluator.number_of_search << std::endl
     << " number of partitioned trees = " << evaluator.number_of_partitioned_tree << std::endl
     << " number of cpu cores = " << evaluator.number_of_cpu_core << std::endl
     << " selectivity = " << evaluator.selectivity << std::endl
     << " query size = " << evaluator.query_size << std::endl;

  return os;
}

} // End of evaluator namespace
} // End of ursus namespace
