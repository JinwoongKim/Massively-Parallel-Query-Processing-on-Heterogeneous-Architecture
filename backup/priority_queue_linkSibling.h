void LinkUpSibling( char* buf, int tree_height, int totalNodes)
{
	long node_off = (totalNodes-1)*BVH_PGSIZE;

	BVH_Node* n = (BVH_Node*) (buf + node_off);

	long sibling[tree_height];
	for( int i=0; i<tree_height; i++)
		sibling[i] = NULL;

	for( int i=0; i<totalNodes; i++, n--)
	{
		//    printf("nodeoff  %lu\n", node_off);
		//    printf("sibling value  %lu\n", sibling[n->getLevel()]);
		long s_value  = sibling[n->getLevel()];
		//    printf("n  %lu\n", n);
		//    printf("anyway  %lu\n", n+BVH_PGSIZE);
		//    printf("anyway-8  %lu\n", n+BVH_PGSIZE-8);
		memcpy(buf+node_off+8+(16+8*NUMDIMS)*NODECARD, &s_value, sizeof(long));
		sibling[n->getLevel()] = node_off;

		//    printf("morton code hIndex %lu\n",n->getMaxMortonCode());
		//    printf("n %lu\n",n);
		//    if( n->getLevel() > 0)
		//    {
		//      printf("# of childs %d\n",n->getCount());
		//      for( int j=0; j<n->getCount(); j++)
		//        printf("%d child %lu\n", j, n->getBranch(j).getChild());
		//    }
		//    printf("level %d\n",n->getLevel());
		//    printf("\n\n");

		node_off -= BVH_PGSIZE;

	}


}

