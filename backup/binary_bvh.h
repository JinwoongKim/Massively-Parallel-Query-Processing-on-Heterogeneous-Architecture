//Binary BVH tree
int findSplit( struct Branch *data, int first,int  last){
	// Identical Morton codes => split the range in the middle.
	//
	unsigned int firstCode = data[first].hIndex;
	unsigned int lastCode = data[last].hIndex;

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.

	int commonPrefix = __builtin_clz(firstCode ^ lastCode);

	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.

	int split = first; // initial guess
	int step = last - first;

	do
  {
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position

		if (newSplit < last)
		{
			unsigned int splitCode = data[newSplit].hIndex;
			int splitPrefix = __builtin_clz(firstCode ^ splitCode);


			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	}
  while (step > 1);

  return split;
}

Node* generateHierarchy( struct Branch* data,
		int           first,
		int           last)
{
	//Single object => create a leaf node.
	if (first == last){
		struct Node* node = (struct Node*) malloc( sizeof(struct Node));
		node->count=1;
		node->level=0; //leaf node
		node->branch[0].hIndex = data[first].hIndex;
		node->branch[0] = data[first];
		return node;
	}

	// Determine where to split the range.
	int split = findSplit(data, first, last);

	// Process the resulting sub-ranges recursively.

	Node* childA = generateHierarchy(data, first, split);
	Node* childB = generateHierarchy(data, split + 1, last);

	struct Node* node = (struct Node*) malloc( sizeof(struct Node));
	node->count=2;
	node->level=1; //internal node
	node->branch[0].child = childA;
	node->branch[1].child = childB;
	return node;
}
