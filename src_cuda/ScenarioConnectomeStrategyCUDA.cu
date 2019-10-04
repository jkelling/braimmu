#include "ScenarioConnectomeStrategyCUDA.h"

#include "scenario_connectome.h"

#include "cudaError.h"

#include <cuda.h>

#include <algorithm>

using namespace std;

struct Coord {
  int x;
  int y;
  int z;
};

__constant__ ScenarioConnectome::properties prop;
__constant__ int nvl[ndim];
__constant__ ScenarioConnectomeStrategyCUDA::AllocPitch pitch;

static __device__ constexpr int tissue(int type)
{
	return 1<<type;
}

static __global__ void derivativeKernel(const double* agent, double* agent2, const int* type, const ScenarioConnectomeStrategyCUDA::array_properties arr_prop, int nall,
double dt, int step);

static __global__ void updateKernel(double* agent, const double* deriv, const int* type, const ScenarioConnectomeStrategyCUDA::array_properties arr_prop, double dt, int nall);

size_t ScenarioConnectomeStrategyCUDA::nvx()
{
	return m_this->nvl[0]+2;
}
size_t ScenarioConnectomeStrategyCUDA::nvyz()
{
	return (m_this->nvl[1]+2)*(m_this->nvl[2]+2);
}

ScenarioConnectomeStrategyCUDA::ScenarioConnectomeStrategyCUDA(ScenarioConnectome* pthis)
	: ScenarioConnectomeAbstractStrategy(pthis)
{
	const size_t width = this->nvx();
	const size_t height = this->nvyz();
	CUDA_SAFE_CALL(
		cudaMallocPitch(&arr_prop.Dtau, &m_allocPitch.pDouble, width*sizeof(double), height*ndim)
		);
	std::cout << "pitch Dtau: " << m_allocPitch.pDouble << std::endl;
	CUDA_SAFE_CALL(
		cudaMallocPitch(&agent, &m_allocPitch.pDouble, width*sizeof(double), height*ScenarioConnectomeAgents::num_agents)
		);
	std::cout << "pitch agent: " << m_allocPitch.pDouble << std::endl;
	CUDA_SAFE_CALL(
		cudaMallocPitch(&deriv, &m_allocPitch.pDouble, width*sizeof(double), height*ScenarioConnectomeAgents::num_agents)
		);
	std::cout << "pitch deriv: " << m_allocPitch.pDouble << std::endl;
	CUDA_SAFE_CALL(
		cudaMallocPitch(&type, &m_allocPitch.pInt, width*sizeof(int), height)
		);
	std::cout << "pitch type: " << m_allocPitch.pInt << std::endl;

	m_allocPitch.pDouble /= sizeof(double);
	m_allocPitch.pInt /= sizeof(int);

	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(prop, &m_this->prop, sizeof(ScenarioConnectome::properties))
		);
	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(nvl, (void*)m_this->nvl.data(), sizeof(int)*m_this->nvl.size())
		);
	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(pitch, &m_allocPitch, sizeof(AllocPitch))
		);
}

ScenarioConnectomeStrategyCUDA::~ScenarioConnectomeStrategyCUDA()
{
	CUDA_SAFE_CALL(
		cudaFree(arr_prop.Dtau)
		);
	CUDA_SAFE_CALL(
		cudaFree(agent)
		);
	CUDA_SAFE_CALL(
		cudaFree(deriv)
		);
	CUDA_SAFE_CALL(
		cudaFree(type)
		);
}

void ScenarioConnectomeStrategyCUDA::push()
{
	const size_t width = this->nvx();
	const size_t height = this->nvyz();
	for(int a = 0; a < ndim; ++a)
	{
		CUDA_SAFE_CALL(
			/*cudaMemcpy(arr_prop.Dtau+m_this->nall*a, m_this->arr_prop.Dtau[a].data(), m_this->nall*sizeof(double), cudaMemcpyHostToDevice)*/
			cudaMemcpy2D(arr_prop.Dtau+height*m_allocPitch.pDouble*a, m_allocPitch.pDouble*sizeof(double)
				, m_this->arr_prop.Dtau[a].data(), width*sizeof(double), width*sizeof(double), height
				, cudaMemcpyHostToDevice)
			);
	}
	for(int a = 0; a < ScenarioConnectomeAgents::num_agents; ++a)
	{
		CUDA_SAFE_CALL(
			/*cudaMemcpy(agent+m_this->nall*a, m_this->agent[a].data(), m_this->nall*sizeof(double), cudaMemcpyHostToDevice)*/
			cudaMemcpy2D(agent+height*m_allocPitch.pDouble*a, m_allocPitch.pDouble*sizeof(double)
				, m_this->agent[a].data(), width*sizeof(double), width*sizeof(double), height
				, cudaMemcpyHostToDevice)
			);
	}
	CUDA_SAFE_CALL(
		cudaMemsetAsync(type, 0, sizeof(int)*m_allocPitch.pInt*height)
		);
	CUDA_SAFE_CALL(
		/*cudaMemcpy(type, m_this->type.data(), m_this->nall*sizeof(int), cudaMemcpyHostToDevice)*/
		cudaMemcpy2D(type, m_allocPitch.pInt*sizeof(int)
			, m_this->type.data(), width*sizeof(int), width*sizeof(int), height
			, cudaMemcpyHostToDevice)
		);
}

void ScenarioConnectomeStrategyCUDA::pop()
{
	const size_t width = this->nvx();
	const size_t height = this->nvyz();
	for(int a = 0; a < ScenarioConnectomeAgents::num_agents; ++a)
	{
		CUDA_SAFE_CALL(
			/*cudaMemcpy(m_this->agent[a].data(), agent + m_this->nall*a, m_this->nall*sizeof(double), cudaMemcpyDeviceToHost)*/
			cudaMemcpy2D(m_this->agent[a].data(), width*sizeof(double)
				, agent+height*m_allocPitch.pDouble*a, m_allocPitch.pDouble*sizeof(double)
				, width*sizeof(double), height
				, cudaMemcpyDeviceToHost)
			);
		CUDA_SAFE_CALL(
			/*cudaMemcpy(m_this->deriv[a].data(), deriv+m_this->nall*a, m_this->nall*sizeof(double), cudaMemcpyDeviceToHost)*/
			cudaMemcpy2D(m_this->deriv[a].data(), width*sizeof(double)
				, deriv+height*m_allocPitch.pDouble*a, m_allocPitch.pDouble*sizeof(double)
				, width*sizeof(double), height
				, cudaMemcpyDeviceToHost)
			);
	}
}

using namespace ScenarioConnectomeAgents;

/*static __global__ void zeroKernel(double* data, size_t n)
{
	int* tmp = (int*)data;
	const int id = threadIdx.x + blockIdx.x*blockDim.x;
	tmp[id] = 0;
	tmp[id+n] = 0;
}*/

/* ----------------------------------------------------------------------*/
void ScenarioConnectomeStrategyCUDA::derivatives() {

	static constexpr int BLOCK_DIM = 128;
  // set derivatives of all voxels to zero
#if 0
	{
		const size_t n = ScenarioConnectomeAgents::num_agents*m_this->nall;
		zeroKernel<<<n/BLOCK_DIM + (n%BLOCK_DIM>0), BLOCK_DIM>>>(deriv, n);
	}
#endif

  const dim3 blocks(m_this->nvl[0]/BLOCK_DIM + (m_this->nvl[0]%BLOCK_DIM>0), m_this->nvl[1], m_this->nvl[2]);
  derivativeKernel<<<blocks, BLOCK_DIM>>>(agent, deriv, type, arr_prop, m_this->nall,m_this->dt, m_this->step);
  // derivativeKernel<<<m_this->nall/BLOCK_DIM + (m_this->nall%BLOCK_DIM>0), BLOCK_DIM>>>(
  //    agent, deriv, type, arr_prop, m_this->nall,m_this->dt, m_this->step);
  std::swap(agent, deriv);
}

static __device__ int find_id(int i, int j, int k)
{
	return i + (pitch.pDouble) * (j + (nvl[1] + 2) * k);
}

static __device__ Coord find_coord(int i)
{
  Coord coord;
  coord.x = i % (nvl[0] + 2);
  i /= (nvl[0] + 2);
  coord.y = i % (nvl[1] + 2);
  coord.z = i / (nvl[1] + 2);
  
  return coord;

}

static __global__ void derivativeKernel(const double* agent, double* agent2,
    const int* type, const ScenarioConnectomeStrategyCUDA::array_properties arr_prop,
    int nall, double dt, int step)
{
  const int ii = threadIdx.x + blockDim.x*blockIdx.x;
  const int jj = blockIdx.y +1;
  const int kk = blockIdx.z +1;
  nall = pitch.pDouble * (nvl[1]+2)*(nvl[2]+2);
  if(ii > 0 && ii < nvl[0]+1)
  {
    const int i = find_id(ii,jj,kk);

    const int t = type[i];
    if (t & tissue(EMP)) return;

  // direct function or time derivatives

	const auto ag_sAb = agent[sAb * nall + i];
	const auto ag_fAb = agent[fAb * nall + i];
	const auto ag_phr = agent[phr * nall + i];

  double dum = prop.kp * ag_sAb * ag_fAb
				 + prop.kn * ag_sAb * ag_sAb;

  agent2[sAb * nall + i] = ag_sAb + dt* ((t & tissue(CSF))
    // sAb, fAb, and tau efflux from CSF
    ? -prop.es * ag_sAb
    // in parenchyma (WM and GM)
     : agent[neu * nall + i] * agent[cir * nall + i]
      - dum
      - prop.ds * agent[mic * nall + i] * ag_sAb);

  agent2[fAb * nall + i] = ag_fAb + dt* ((t & tissue(CSF))
    ? -prop.es * ag_fAb
    : dum
      - prop.df * agent[mic * nall + i] * ag_fAb);

  dum = prop.ktau * ag_phr;
  // tau protein phosphorylation due to fAb and neu
  agent2[phr * nall + i] = ag_phr + dt* ((t & tissue(CSF))
    ? -prop.ephi * ag_phr
    : prop.kphi * ag_fAb * agent[neu * nall + i]
      - dum);

  // tau tangle formation from phosphorylated tau
  agent2[tau * nall + i] = agent[tau * nall + i] + dt* ((t & tissue(CSF)) ? 0 : dum) ;

  // neuronal death due to tau aggregation
  agent2[neu * nall + i] = agent[neu * nall + i] + dt* ((t & tissue(CSF)) ? 0 : -prop.dnt * agent[tau * nall + i] * agent[neu * nall + i]);

  // astrogliosis
  dum = ag_fAb * agent[mic * nall + i];
  agent2[ast * nall + i] = agent[ast * nall + i] + dt* ((t & tissue(CSF)) ? 0 : prop.ka * (dum / (dum + prop.Ha) - agent[ast * nall + i]));

  // circadian rhythm
  // !!inverse CSF check!!
  agent2[cir * nall + i] = agent[cir * nall + i] + dt* ((!(t & tissue(CSF)) && (prop.c_cir > 0))
    ? - prop.C_cir * prop.c_cir * prop.omega_cir
      * sin(prop.omega_cir * dt * step)
    : 0);

      double de_mic = 0.;
      const double ag_mic = agent[mic * nall + i];
      #pragma unroll
		  for(int s = -1; s <= 1; s+=2)
			  for (int d=0; d < 3; d+=1) {
			    const int j = find_id(ii +s*(d==0),jj +s*(d==1),kk +s*(d==2));

			    if (type[j] & tissue(EMP)) continue;

			    const double del_phr = ag_phr - agent[phr * nall + j];

			    // diffusion of tau
			    agent2[phr * nall + i] -= 0.5 * (arr_prop.Dtau[ nall * d + i] + arr_prop.Dtau[nall * d + j]) * del_phr *dt;

			    const double del_sAb = ag_sAb - agent[sAb * nall + j];

			    // diffusion of sAb
			    agent2[sAb * nall + i] -= prop.D_sAb * del_sAb *dt;

			    // only in parenchyma
			    if (t & tissue(WM) || t & tissue(GM))
				  if (type[j] & tissue(WM) || type[j] & tissue(GM)) {
				    const double del_fAb = ag_fAb - agent[fAb * nall + j];
				    const double del_mic = ag_mic - agent[mic * nall + j];

				    // migration of microglia toward higher sAb concentrations
				    de_mic += prop.cs * del_sAb * ((del_sAb > 0.0) ? agent[mic * nall + j] : ag_mic);

				    // migration of microglia toward higher fAb concentrations
            de_mic += prop.cf * del_fAb * ((del_fAb > 0.0) ? agent[mic * nall + j] : ag_mic);

				    // diffusion of microglia
				    de_mic -= prop.D_mic * del_mic;
		      }
		    }
		agent2[mic * nall + i] = ag_mic + de_mic *dt;
	  }
}

#include "ScenarioConnectomeStrategyCUDANewton.cu"
