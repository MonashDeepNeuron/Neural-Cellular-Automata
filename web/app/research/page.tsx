export default function Research() {
	return (
		<>
			{/* Main Content */}
			<div className='centre'>
				<h1>Neural Cellular Automata</h1>
				<h3>- Top 5 papers, summarised!</h3>
				<p>
					<i>Last updated Dec 2024</i>
				</p>

				<h2>Growing Neural Cellular Automata</h2>
				<p>
					Differentiable Model of Morphogenesis, <i>Feb. 11, 2020</i>. doi:
					<a href='https://doi.org/10.23915/distill.00023'> 10.23915/distill.00023</a>
				</p>
				<h4>
					<i>Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, Michael Levin</i>
				</h4>
				<p>
					This is the foundational paper behind Neural Cellular Automata (NCA), which proposes a minimalistic model that can produce
					organism-like behaviour. This model is able to grow into simple but irregular images and demonstrates robustness against
					perturbation.
				</p>
				<p>
					The structure and techniques used in this paper are fundamental in future work, including:
					<ul>
						<li>
							<b>Convolutional Network architecture</b> using a learnable or non-learnable 3x3 convolution.
						</li>
						<li>
							<b>Residual neural network architecture</b> for better training and gradient calculation.
						</li>
						<li>
							<b>Stochastic updates</b> (random cells updated per iteration) for robustness.
						</li>
						<li>
							<b>Usage of hidden channels</b> for recurrent neural network-like behaviour.
						</li>
					</ul>
				</p>

				<h2>Self-Organising Textures</h2>
				<p>
					Neural Cellular Automata Model of Pattern Formation, <i>Feb. 11, 2021</i>. doi:
					<a href='https://doi.org/10.23915/distill.00027.003'> 10.23915/distill.00027.003</a>
				</p>
				<h4>
					<i>Eyvind Niklasson, Alexander Mordvintsev, Ettore Randazzo, Michael Levin</i>
				</h4>
				<p>
					This paper advances NCA by focusing on local, small-scale features rather than generating an entire image. The result allows the
					model to generate cohesive patterns across grids of any size.
				</p>
				<p>Below are the results from our own re-implementation:</p>

				<div className='sideBySide'>
					<div className='sideDiagramContainer'>
						<video className='sideDiagram' loop autoPlay muted>
							<source src='/Pages/Images/nca_output.mp4' type='video/mp4' />
						</video>
						<p>
							<i>Above: Screen record of NCA algorithm in live operation after training.</i>
						</p>
					</div>
					<div className='sideDiagramContainer'>
						<img src='/Pages/Images/knit.jpg' className='sideDiagram' alt='Target Knitted Texture' />
					</div>
				</div>

				<h2>Mesh Neural Cellular Automata</h2>
				<p>
					<i>Jul. 19, 2024</i>. doi:
					<a href='https://doi.org/10.1145/3658127'> 10.1145/3658127</a>
				</p>
				<h4>
					<i>Ehsan Pajouheshgar, Yitao Xu, Alexander Mordvintsev, Eyvind Niklasson, Tong Zhang, Sabine Süsstrunk</i>
				</h4>
				<p>
					This paper focuses on 3D texture generation, moving NCA onto a mesh or graph. Each cell is now a node, with its neighbours defined
					by the connections between graph edges.
				</p>

				<h2>Further Work</h2>
				<p>
					Further research has explored NCA in multiple areas, demonstrating capabilities such as:
					<ul>
						<li>
							<b>Combining multiple algorithms</b>
						</li>
						<li>
							<b>Adapting to different grid sizes without explicit retraining</b>
						</li>
						<li>
							<b>Classifying images</b>
						</li>
						<li>
							<b>Interpreting external signals</b>
						</li>
						<li>
							<b>Producing exceptionally small and efficient models</b>
						</li>
					</ul>
				</p>
			</div>
		</>
	);
}
