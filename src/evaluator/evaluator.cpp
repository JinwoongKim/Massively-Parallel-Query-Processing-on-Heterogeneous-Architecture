#include "evaluator/evaluator.h"

#include "tree/hybrid.h"

#include <cassert>
#include <unistd.h>

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

  // Parse the args and initialize variables
  if( !ParseArgs(argc, argv)) {
    PrintHelp(argv);
    return false;
  }

  // Read dataset based on initialized variables
  ReadDataSet();

  return true;
}

//TODO :: Need to fix?  scrub
//TODO :: paint it up
void Evaluator::PrintHelp(char **argv) const {
  std::cerr << "Usage:\n" << *argv << std::endl << 
  " -d number of data\n" 
  " [ -q number of queries, default : 0 (debugging mode) ]\n" 
  " [ -i index type, default : R-trees]\n"
  " [ -m search algorithm type, 1: MPES, 2: MPTS, 3: MPHR, 4: MPHR2\n \
  5: Short-Stack, 6: Parent-Link, 7: Skip-Pointer ]\n"
  " [ -o distribution policy, default : braided version]\n"
  " [ -b braided version, number of block, default : 128 ]\n"
  " [ -p partitioned version, number of block ]\n" 
  " [ -s selection ratio(%), default : 1 (%) ]\n"
  " [ -c number of cpu cores, default : 1 ]\n" 
  " [ -w workload offset, default : 0 ]\n" 
  " [ -v Specified device(GPU) id, default : 0 ]\n" 
  "\n e.g:   ./cuda -d 1000000 -q 1000 -s 0.5 -c 4 -w 3\n" 
  << std::endl;
}


//TODO :: Boost.Program_options
// http://www.boost.org/doc/libs/1_59_0/doc/html/program_options.html
bool Evaluator::ParseArgs(int argc, char **argv)  {

  static const char *options="d:D:q:Q:p:P:b:B:s:S:c:C:w:W:o:O:i:I:m:M:k:K:v:V:";
  //extern char *optarg;
  std::string number_of_data_str;
  int current_option;

  while ((current_option = getopt(argc, argv, options)) != -1) {
    switch (current_option) {
//      case 'v':
//      case 'V': DEVICE_ID = atoi(optarg); break;
//      case 'k':
//      case 'K': keepDoing = atoi(optarg); break;
//      case 'w':
//      case 'W': WORKLOAD = atoi(optarg); break;
//      case 'o':
//      case 'O': POLICY = atoi(optarg); break;
//      case 'i':
//      case 'I': BUILD_TYPE = atoi(optarg); break;
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
      case 'Q': number_of_searches = atoi(optarg); break;
      case 'p':
      case 'P': number_of_partitioned_trees = atoi(optarg); break;
      case 's':
      case 'S': selectivity = std::string(optarg);  break;
      case 'c':
      case 'C': number_of_cpu_cores = atoi(optarg); break;
     default: break;
    } // end of switch
  } // end of while

  if( number_of_data_str.empty() ) return false;

  if( number_of_cpu_cores > 0 )
    number_of_cpu_cores = ( number_of_partitioned_trees > 1 ) ? number_of_partitioned_trees : 1;  

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

  // TODO Hard coded now
  tree::Tree *hybrid = new tree::Hybrid();
  trees.push_back(hybrid);

//  if( METHOD[7] == true)
//    METHOD[0] = METHOD[1] = METHOD[2] =  METHOD[3] = METHOD[4] = METHOD[5] = METHOD[6] = true;
//
//
//  if( (METHOD[0] || METHOD[1] || METHOD[2] ||  METHOD[3] || METHOD[4] || METHOD[5] || METHOD[6]) && number_of_searches == 0)
//  {
//    number_of_searches = 1000;
//  }

//  if(BUILD_TYPE == 0)  {
//    printf("DATATYPE : %s, PGSIZE %d, NUMDIMS %d, number_of_thread_blocks %d, NUMTHREADS %d,  NODECARD %d, number_of_data %d, number_of_searches %d, SELECTION RATIO %s, NCPU %d,   number_of_partitioned_trees %d, ",
//            DATATYPE,      PGSIZE,    NUMDIMS,    number_of_thread_blocks,    NUMTHREADS,     NODECARD,    number_of_data,    number_of_searches,    SELECTIVITY ,       number_of_cpu_cores, number_of_partitioned_trees       );
//  }
//  else
//  {
//    printf("DATATYPE : %s, PGSIZE %d, NUMDIMS %d, number_of_thread_blocks %d, NUMTHREADS %d,  NODECARD %d, number_of_data %d, number_of_searches %d, SELECTION RATIO %s, NCPU %d,   number_of_partitioned_trees %d, ",
//            DATATYPE,      BVH_PGSIZE,    NUMDIMS,    number_of_thread_blocks,    NUMTHREADS,     NODECARD,    number_of_data,    number_of_searches,    SELECTIVITY ,       number_of_cpu_cores, number_of_partitioned_trees       );
//  }
//
//  if( BUILD_TYPE == 0 )
//    printf("\n\nRTrees will be build up.. \n");
//  else if ( BUILD_TYPE == 1 || BUILD_TYPE == 2 )
//    printf("\n\nBVH-Trees (TYPE : %d )  will be build up..\n", BUILD_TYPE);
//  else
//    printf("\n\nHilbertRadix Tree (TYPE : %d )  will be build up..\n", BUILD_TYPE);
//
//
//  if( POLICY == 0 )
//    printf("Original distribution\n");
//  else
//    printf("Roundrobin distribution\n");
//

  // TODO :: REMOVE, just for debugging now
  std::cout << *this << std::endl;
  return true;
}


/**
 * @brief Build all indexing structures in tree_queue with input_dataset
 * @return true if building all indexing structures successfully,
 *  otherwise return false 
 */
bool Evaluator::Build(void) {
  for(auto tree : trees) {
    if(!tree->Build(input_data_set)) {
      return false;
    }
  }
  return true;
}

bool Evaluator::ReadDataSet(void){
  // Read data set
  //TODO : hard coded now...
  input_data_set.reset ( new io::DataSet(GetNumberOfDims(), number_of_data,
                         "/home/jwkim/dataFiles/input/real/NOAA0.bin",
                         DATASET_TYPE_BINARY)); 
  return true;
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const Evaluator &evaluator) {
  os << " Evaluator : " << std::endl
     << " number of data = " << evaluator.number_of_data << std::endl
     << " number of degrees = " << GetNumberOfDegrees() << std::endl
     << " number of thread blocks = " << GetNumberOfBlocks() << std::endl
     << " number of threads = " << GetNumberOfThreads() << std::endl
     << " number of searches = " << evaluator.number_of_searches << std::endl
     << " number of partitioned trees = " << evaluator.number_of_partitioned_trees << std::endl
     << " number of cpu cores = " << evaluator.number_of_cpu_cores << std::endl
     << " selectivity = " << evaluator.selectivity << std::endl
     << " query size = " << evaluator.query_size << std::endl;

  return os;
}

} // End of evaluator namespace
} // End of ursus namespace
