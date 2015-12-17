// :: TO DO
// :: DumpToFile, LoadToFile
// :: every search function(host)
// :: run to mpi 
// :: distribution policy on multi-GPUs or partitioned version.


//#define TP
#define NONILP
//#define ILP
//#define MPI

//MPI
#ifdef MPI
#include <mpi.h>
#endif

#include "index.h"

#include "bvh.h"
#include "radix.h"
#include "host.h"
#include "rtree.h"

#include "mpes.h"
#include "mpts.h"

#include "mphr.h"

#ifndef TP
#if NUMDIMS < 64
#include "shortstack.h"
#endif
#endif

#include "parentLink.h"
#include "skippointer.h"

int main(int argc, char *argv[])
{
  //	 cudaProfilerInitialize(); 


  //Display selected parallelism
#ifdef TP
  printf("Task Parallelism\n");
#else
#ifdef ILP
  printf("Data Parallelism(ILP)\n");
#else
  printf("Data Parallelism(No-ILP)\n");
#endif
#endif


  // To find an available gpu card
  //  if( find_an_available_gpu(8) == -1 )
  //		return 0;

  //Parsing for a given arguments
  if ( !ParseArgs(argc, argv) || NUMDATA == 0 ){
    printHelp(argv);
    return -1;
  }

  cudaSetDevice(DEVICE_ID);
  printf("GPU ( %d ) \n", DEVICE_ID);

  number_of_procs=1;
#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif

  for( int index_type=0; index_type<3; index_type++) 
  {
    //  index_type == 0  build an index structure for MPES, MPTS, MPHR, MPHR2, ShortStack
    //  index_type == 1  build an index structure for ParentLink
    //  index_type == 2  build an index structure for SkipPointer

    if( index_type == 0 && !(METHOD[0] || METHOD[1] || METHOD[2] || METHOD[3] || METHOD[4])) continue;
    if( index_type == 1 && (!METHOD[5] )) continue;
    if( index_type == 2 && (!METHOD[6] )) continue;

    //load index from file or build index
    //if( !TreeLoadFromFile())

    //BVH build or R-tree build
    printf("Start Build Index...\n");

    //build an index structure(R-trees, BVH, or Radix
    //Treees) according to a index type

    if( BUILD_TYPE == 0 )
      Build_Rtrees(index_type);
    else if(BUILD_TYPE == 1 || BUILD_TYPE == 2)
      Build_BVH(index_type);
    else
      Build_RadixTrees(index_type);

    printf("####################################################\n");
    printf("############### SEARCH THE DATA ####################\n");
    printf("####################################################\n");

#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    printf("Start Search Operation\n");

    if( NUMSEARCH )
    {
      if( NCPUCORES )
      {
        //BUILD_TYPE == 0, R-trees
        //BUILD_TYPE == 1, BVH
        if( BUILD_TYPE == 0)
          RTree_Multicore();
        else
          BVHTree_Multicore();
      }

      // profiling a cuda program from here
      cudaProfilerStart();	

      if( index_type == 0 )
      {
        // MPES
        if( METHOD[0] == true )
        {
          printf("MPES start\n");
          MPES(number_of_procs, myrank);
#ifdef MPI
          MPI_Barrier(MPI_COMM_WORLD);
#endif
          cout << endl << endl;
        }

        // MPTS, it only works for balanced tree structures
        if( METHOD[1] == true && BUILD_TYPE == 0 )  
        {
          if( BUILD_TYPE == 0)
            MPTS(number_of_procs, myrank);
          else
            printf("We could not build BVH-trees for MPTS\n");

#ifdef MPI
          MPI_Barrier(MPI_COMM_WORLD);
#endif
          cout << endl << endl;
        }

        // MPHR
        if( METHOD[2] == true )
        {
          printf("MPHR start\n");
          MPHR(number_of_procs, myrank); 
#ifdef MPI
          MPI_Barrier(MPI_COMM_WORLD);
#endif
          cout << endl << endl;
        }


        // MPHR2 (=MPRS)
        if( METHOD[3] == true )
        {
          //if( keepDoing ){
          //	while( scanf("%d",&keepDoing), keepDoing == 1 ){
          //		MPHR2(number_of_procs, myrank); 
          //	}
          //}
          //else
          //{
          MPHR2(number_of_procs, myrank); 
          //}

#ifdef MPI
          MPI_Barrier(MPI_COMM_WORLD);
#endif
          cout << endl << endl;
        }

#ifndef TP
#if NUMDIMS < 64
        // ShortStack
        /*
           if( METHOD[4] == true )
           {
           shortstack(number_of_procs, myrank); 
           MPI_Barrier(MPI_COMM_WORLD);
           cout << endl << endl;
           }
         */
#endif
#endif
      }
      /*
         else if( index_type == 1)
         {
      // ParentLink
      if(METHOD[5] == true)
      {
      ParentLink(number_of_procs, myrank); 
#ifdef MPI
MPI_Barrier(MPI_COMM_WORLD);
#endif
cout << endl << endl;
}
}
else
{
      // SkipPointer
      if( METHOD[6] == true)
      {
      skippointer(number_of_procs, myrank); 
#ifdef MPI
MPI_Barrier(MPI_COMM_WORLD);
#endif
cout << endl << endl;
}
}
       */

}
cout << "END" << endl;


//if( BUILD_TYPE == 1 || (BUILD_TYPE == 0 && index_type > 0 ))
//	globalFreeDeviceBVHRoot<<<1,1>>>(PARTITIONED );
//
//	 else
//	 globalFreeDeviceRoot<<<1,1>>>(PARTITIONED );
//	 cudaThreadSynchronize();


if( METHOD[0] || METHOD[1] || METHOD[2] || METHOD[3] || METHOD[4] || METHOD[5] || METHOD[6] )
{
  FILE *fp = fopen("elapsed_time.log", "a");
  FILE *fp2 = fopen("visited_nodes.log", "a");
  if(  index_type == 0 )
  {
    if(BUILD_TYPE == 0)
    {
      fprintf(fp, "DATATYPE : %s, PGSIZE %d, NUMDIMS %d, NUMBLOCK %d, NUMTHREADS %d,  NODECARD %d, NUMDATA %d, NUMSEARCH %d, SELECTION RATIO %s, NCPU %d,   PARTITIONED %d, ",
          DATATYPE,      PGSIZE,    NUMDIMS,    NUMBLOCK,    NUMTHREADS,     NODECARD,    NUMDATA,    NUMSEARCH,    SELECTIVITY ,       NCPUCORES, PARTITIONED       );
      fprintf(fp2, "DATATYPE : %s, PGSIZE %d, NUMDIMS %d, NUMBLOCK %d, NUMTHREADS %d,  NODECARD %d, NUMDATA %d, NUMSEARCH %d, SELECTION RATIO %s, NCPU %d,   PARTITIONED %d, ",
          DATATYPE,      PGSIZE,    NUMDIMS,    NUMBLOCK,    NUMTHREADS,     NODECARD,    NUMDATA,    NUMSEARCH,    SELECTIVITY ,       NCPUCORES, PARTITIONED       );
    }
    else
    {
      fprintf(fp,"DATATYPE : %s, PGSIZE %d, NUMDIMS %d, NUMBLOCK %d, NUMTHREADS %d,  NODECARD %d, NUMDATA %d, NUMSEARCH %d, SELECTION RATIO %s, NCPU %d,   PARTITIONED %d, ",
          DATATYPE,      BVH_PGSIZE,    NUMDIMS,    NUMBLOCK,    NUMTHREADS,     NODECARD,    NUMDATA,    NUMSEARCH,    SELECTIVITY ,       NCPUCORES, PARTITIONED       );
      fprintf(fp2,"DATATYPE : %s, PGSIZE %d, NUMDIMS %d, NUMBLOCK %d, NUMTHREADS %d,  NODECARD %d, NUMDATA %d, NUMSEARCH %d, SELECTION RATIO %s, NCPU %d,   PARTITIONED %d, ",
          DATATYPE,      BVH_PGSIZE,    NUMDIMS,    NUMBLOCK,    NUMTHREADS,     NODECARD,    NUMDATA,    NUMSEARCH,    SELECTIVITY ,       NCPUCORES, PARTITIONED       );
    }

    if( BUILD_TYPE == 0 )
    {
      fprintf(fp,"\n\nRTrees will be build up.. \n");
      fprintf(fp2,"\n\nRTrees will be build up.. \n");
    }
    else
    {
      fprintf(fp,"\n\nBVH-Trees (TYPE : %d )  will be build up..\n", BUILD_TYPE);
      fprintf(fp2,"\n\nBVH-Trees (TYPE : %d ) will be build up..\n", BUILD_TYPE);
    }
  }

  for( int i=0; i<7; i++)
  {
    if( i == 1 && (BUILD_TYPE > 0 )) // skip when build BVH-tree and MPTS together
      continue;
    else if( METHOD[i] )
      fprintf(fp,"%.6f \n",t_time[i] );
  }


  for( int i=0; i<7; i++)
  {
    if( METHOD[i] )
    {
      //					fprintf(fp2,"%.6f\n", t_visit[i] );
      //					fprintf(fp2,"%.6f\n", t_rootVisit[i] );
      //					fprintf(fp2,"%.6f\n", t_pop[i] );
      //					fprintf(fp2,"%.6f\n", t_push[i] );
      //					fprintf(fp2,"%.6f\n", t_parent[i] );
      //					fprintf(fp2,"%.6f\n", t_skipPointer[i] );
      float t_totalVisit = t_visit[i] + t_rootVisit[i] + t_pop[i] + t_parent[i] + t_skipPointer[i];
      fprintf(fp2,"%.6f\n", t_totalVisit );
    }
  }
}
}

#ifdef MPI
MPI_Finalize();
#endif
return 0;
}
