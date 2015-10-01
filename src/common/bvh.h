class BVH_Node {
public:
//	int next_visit[128]; //number of blocks
	int count;
	int level;
	BVH_Branch branch[NODECARD];
	BVH_Node* sibling;
	BVH_Node* parent;
public:
	BVH_Node()
	{
	}
	BVH_Node(const int _count, const int _level)
	{
		count = _count;
		level = _level;
		for(int i=0; i<NODECARD; i++)
		{
			branch[i].child = NULL;
			sibling = NULL;
			parent = NULL;
		}
	}
	~BVH_Node()
	{
	}
	void setLevel(int _level)
	{
		level = _level;
	}
	void setCount(int _count)
	{
		count = _count;
	}
	
	int getLevel()
	{
		return level;
	}
	int getCount()
	{
		return count;
	}
	BVH_Branch getBranch(int i)
	{
		return branch[i];
	}
	float getBranchRectBoundary(int c, int d)
	{
		return branch[c].getRectBoundary(d);
	}
	void setBranchRectBoundary(int c, int d, float boundary )
	{
		branch[c].setRectBoundary(d,boundary);
	}

	void setBranch(int i, BVH_Branch b)
	{
		branch[i] = b;
	}
	void setBranchChild(int i, BVH_Node* child)
	{
		branch[i].setChild(child);
	}
	void setBranchMortonCode(int i, unsigned long long m)
	{
		branch[i].setMortonCode(m);
	}
	unsigned long long getBranchMortonCode(int i)
	{
		return branch[i].getMortonCode();
	}
	unsigned long long getMaxMortonCode()
	{
		return branch[count-1].getMortonCode();
	}
}
