#include <mpi.h>
#include "math.h"

#include "pointers.h"
#include "brain.h"
#include "input.h"
#include "memory.h"
#include "output.h"

#include <algorithm>

using namespace std;
using namespace brain_NS;

/* ----------------------------------------------------------------------*/
Brain::Brain(int narg, char **arg, int rk, int np) {
  me = rk;
  nproc = np;

  MPI_Comm_split(MPI_COMM_WORLD,0,me,&world);

  allocations();

  if (!me)
    printf("Reading input, setup the system ... \n");
  input->file(arg[1], this);

  // output initial step
  //if (!me)
  //  printf("Writing output for the initial step ... \n");
  //output->lammpstrj(this);

  if (output->do_dump)
    output->dump(this);

  if (output->severy > 0)
    output->statistics(this);

  if (!me)
    printf("Integration started. \n");
  integrate(Nrun);

  //printf("proc %i: xlo = %g \n", me, xlo);
  //MPI_Barrier(MPI_COMM_WORLD);
  //printf("proc %i here3 \n",brn->me);
  //if (brn->me == 2)
    //printf("proc %i: unpack itag = %li, c=%i \n",brn->me,itag,c-1);

}

/* ----------------------------------------------------------------------*/
Brain::~Brain() {
  destroy();

  delete region;
  delete output;
  delete comm;
  delete init;
  delete input;
  delete memory;
}

/* ----------------------------------------------------------------------*/
void Brain::allocations() {
  nvoxel = 0;
  nlocal = nghost = nall = 0;
  step = Nrun = 0;
  Nlog = 1000;

  dt = 0.0;
  nevery = -1;
  vlen = vlen_1 = vlen_2 = 0.0;
  vvol = vvol_1 = 0.0;

  for (int ag_id=0; ag_id<num_agents; ag_id++)
    init_val[ag_id] = -1.0;

  D_sAb = diff_sAb = 0.0;
  D_mic = diff_mic = 0.0;
  cs = sens_s = cf = sens_f = 0.0;
  kp = kn = 0.0;
  ds = df = 0.0;
  es = 0.0;

  memory = new Memory();
  input = new Input();
  init = new Init();
  comm = new Comm();
  output = new Output();
  region = new Region();

  nim = NULL;
}

/* ----------------------------------------------------------------------*/
void Brain::destroy() {
  memory->destroy(x);
  memory->destroy(tag);

  memory->destroy(num_neigh);
  memory->destroy(neigh);

  //memory->destroy(agent);
  //memory->destroy(grad);

  if(nim)
    nifti_image_free(nim);

}

void Brain::dump_mri(const vector<string> &arg)
{
	// get partitions
	std::vector<std::array<std::pair<int, int>, 3>> xlohi(nproc);
	std::vector<int> I_AGENTS;
	{
		std::vector<std::pair<int,int>> tmp;
		tmp.reserve(num_agents);
		for(int a = 0; a < num_agents; ++a)
		{
			const auto it = std::find(arg.begin(), arg.end(), ag_str[a]);
			if(it != arg.end())
				tmp.push_back(std::make_pair(std::distance(arg.begin(), it)-2, a));
		}
		std::sort(tmp.begin(), tmp.end());
		I_AGENTS.reserve(tmp.size());
		for(const auto& s: tmp)
			I_AGENTS.push_back(s.second);
		printf("I_AGENTS %d %d\n", I_AGENTS.size(), tmp.size());
	}

	{
		std::array<std::pair<int, int>, 3> tmp;
		for(int a = 0; a < 3; ++a)
		{
			tmp[a].first = xlo[a];
			tmp[a].second = xhi[a];
		}
		MPI_Gather((int*)tmp.data(),6,MPI_INT,xlohi.data()
			, 6, MPI_INT,0,world);
	}

	// get sizes
	std::vector<int> rcounts(nproc), displs(nproc, 0);
	std::vector<int> g_type;
	std::vector<std::vector<double>> g_agents(I_AGENTS.size());
	{
		MPI_Gather(&nall,1,MPI_INT
			,rcounts.data(),1,MPI_INT,0,world);
		for(int a = 1; a < displs.size(); ++a)
		{
			displs[a] = displs[a-1] + rcounts[a-1];
		}

		const size_t size = displs.back() + rcounts.back();
		if (!me)
		{
			g_type.resize(size);
			for(auto& a : g_agents)
				a.resize(size);
		}
	}

	// get data
  MPI_Gatherv(type.data(),nall,MPI_INT,g_type.data()
			,rcounts.data(),displs.data(),MPI_INT,0,world);
	for(int a = 0; a < g_agents.size(); ++a)
	{
		const int AGENT = I_AGENTS[a];
		MPI_Gatherv(agent[AGENT].data(),nall,MPI_DOUBLE
				,g_agents[a].data()
				,rcounts.data(),displs.data(),MPI_DOUBLE,0,world);
	}

	
  if (!me)
	{
    nifti_image *nim;

    // if (dsize > 4)
		const int dims5[] = {5, nv[0], nv[1], nv[2], 1, g_agents.size()+1, 1, 1};
		nim = output->nifti_image_setup(this,arg, dims5, NIFTI_INTENT_VECTOR);

		const int I_TYPE = I_AGENTS.size();

    float* data = (float*) nim->data;
		for(int p = 0; p < nproc; ++p)
		{
			const int K = xlohi[p][2].second - xlohi[p][2].first;
			const int J = xlohi[p][1].second - xlohi[p][1].first;
			const int I = xlohi[p][0].second - xlohi[p][0].first;
			int* ptype = g_type.data() + displs[p];
			for (int k=1; k<=K; ++k)
				for (int j=1; j<=J; ++j)
					for (int i=1; i<=I; ++i) {
						const size_t sidx = i + I * (j + J*k);

						const size_t ii = i-1+xlohi[p][0].first;
						const size_t jj = j-1+xlohi[p][1].first;
						const size_t kk = k-1+xlohi[p][2].first;

            const size_t cnim = ii + nim->nx * ( jj
								+ nim->ny * (kk
									+ nim->nz * ( I_TYPE ) ) );

						data[cnim] = ptype[sidx];
					}

			for(int a = 0; a < g_agents.size(); ++a)
			{
				double* pagent = g_agents[a].data() + displs[p];
				for (int k=1; k<=K; ++k)
					for (int j=1; j<=J; ++j)
						for (int i=1; i<=I; ++i) {
							const size_t sidx = i + I * (j + J*k);

							const size_t ii = i-1+xlohi[p][0].first;
							const size_t jj = j-1+xlohi[p][1].first;
							const size_t kk = k-1+xlohi[p][2].first;

							const size_t cnim = ii + nim->nx * ( jj
									+ nim->ny * (kk
										+ nim->nz * ( I_AGENTS[a] ) ) );

							data[cnim] = ptype[sidx];
						}
			}
		}

    nifti_image_write(nim);
    nifti_image_free(nim);
    nim = nullptr;
	}
}
