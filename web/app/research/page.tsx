export default function Research() {
	return (
		<div className="max-w-4xl mx-auto px-6 py-10 text-gray-800">
			{/* Main Content */}
			<h1 className="text-4xl font-bold text-purple-700 mb-4 text-center">Neural Cellular Automata</h1>
			<h3 className="text-lg text-purple-500 text-center mb-2">- Top 5 Papers, Summarised!</h3>
			<p className="text-center italic">Last updated Dec 2024</p>

			{/* Paper 1 */}
			<section className="mb-8">
				<h2 className="text-2xl font-semibold text-purple-600 mt-6">Growing Neural Cellular Automata</h2>
				<p>
					Differentiable Model of Morphogenesis, <i>Feb. 11, 2020</i>. doi:
					<a href="https://doi.org/10.23915/distill.00023" className="text-purple-500 hover:underline"> 10.23915/distill.00023</a>
				</p>
				<h4 className="italic text-gray-600">Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, Michael Levin</h4>
				<p>
					This is the foundational paper behind Neural Cellular Automata (NCA), proposing a minimalistic model that produces organism-like
					behaviour. It demonstrates robustness against perturbation and the ability to grow irregular images.
				</p>
				<p>The structure and techniques used in this paper are fundamental in future work, including:</p>
				<ul className="list-disc list-inside ml-4">
					<li><b>Convolutional Network architecture:</b> Learnable or non-learnable 3x3 convolution.</li>
					<li><b>Residual neural network architecture:</b> Improves training and gradient calculation.</li>
					<li><b>Stochastic updates:</b> Random cells updated per iteration for robustness.</li>
					<li><b>Hidden channels:</b> Enables recurrent neural network-like behaviour.</li>
				</ul>
			</section>

			{/* Paper 2 */}
			<section className="mb-8">
				<h2 className="text-2xl font-semibold text-purple-600">Self-Organising Textures</h2>
				<p>
					Neural Cellular Automata Model of Pattern Formation, <i>Feb. 11, 2021</i>. doi:
					<a href="https://doi.org/10.23915/distill.00027.003" className="text-purple-500 hover:underline"> 10.23915/distill.00027.003</a>
				</p>
				<h4 className="italic text-gray-600">Eyvind Niklasson, Alexander Mordvintsev, Ettore Randazzo, Michael Levin</h4>
				<p>
					This paper advances NCA by focusing on local, small-scale features rather than generating entire images, allowing models to
					generate cohesive patterns across any grid size.
				</p>
				<p>Below are the results from our re-implementation:</p>

				<div className="flex flex-wrap gap-4">
					<div>
						<video className="w-full rounded-md shadow" loop autoPlay muted>
							<source src="/Pages/Images/nca_output.mp4" type="video/mp4" />
						</video>
						<p className="text-sm text-center italic">NCA algorithm in live operation after training.</p>
					</div>
					<div>
						<img src="/Pages/Images/knit.jpg" className="w-full rounded-md shadow" alt="Target Knitted Texture" />
					</div>
				</div>
			</section>

			{/* Paper 3 */}
			<section className="mb-8">
				<h2 className="text-2xl font-semibold text-purple-600">Mesh Neural Cellular Automata</h2>
				<p>
					<i>Jul. 19, 2024</i>. doi:
					<a href="https://doi.org/10.1145/3658127" className="text-purple-500 hover:underline"> 10.1145/3658127</a>
				</p>
				<h4 className="italic text-gray-600">Ehsan Pajouheshgar, Yitao Xu, Alexander Mordvintsev, Eyvind Niklasson, Tong Zhang, Sabine Süsstrunk</h4>
				<p>
					This paper focuses on 3D texture generation, moving NCA onto a mesh or graph. Each cell is now a node, with its neighbours defined
					by graph connections.
				</p>
			</section>

			{/* Further Work */}
			<section className="mb-8">
				<h2 className="text-2xl font-semibold text-purple-600">Further Work</h2>
				<p>Further research has explored NCA in multiple areas, demonstrating capabilities such as:</p>
				<ul className="list-disc list-inside ml-4">
					<li><b>Combining multiple algorithms</b></li>
					<li><b>Adapting to different grid sizes without retraining</b></li>
					<li><b>Classifying images</b></li>
					<li><b>Interpreting external signals</b></li>
					<li><b>Producing small and efficient models</b></li>
				</ul>
			</section>
		</div>
	);
}
