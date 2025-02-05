export default function Intro() {
	return (
		<div className="max-w-4xl mx-auto px-6 py-10 text-gray-800">
			{/* Main Title */}
			<h1 className="text-4xl font-bold text-purple-700 mb-6 text-center">
				Neural Cellular Automata
			</h1>

			{/* Introduction Section */}
			<section className="mb-8">
				<h2 className="text-2xl font-semibold text-purple-600 mb-3">Introduction</h2>
				<p className="text-lg leading-7">
					Neural Cellular Automata (NCA) represent an innovative intersection between traditional cellular automata and neural networks.
					They are capable of learning complex behaviors through simple, local interactions between cells, mimicking biological growth and
					regeneration processes.
				</p>
			</section>

			{/* Cellular Automata Perspective */}
			<section className="mb-8">
				<h2 className="text-2xl font-semibold text-purple-600 mb-3">The Cellular Automata Perspective</h2>
				<p className="text-lg leading-7">
					From the perspective of cellular automata, NCAs build on the concept of simple, rule-based systems where each cell updates its
					state based on the states of its neighbors. This approach enables the emergence of complex patterns from basic rules.
				</p>
			</section>

			{/* Neural Network Perspective */}
			<section>
				<h2 className="text-2xl font-semibold text-purple-600 mb-3">The Neural Network Perspective</h2>
				<p className="text-lg leading-7">
					When viewed through the lens of neural networks, NCAs leverage the power of deep learning to optimize the update rules. This
					allows the system to adapt, learn, and generalize behaviors across various environments, making NCAs versatile tools for modeling
					dynamic systems.
				</p>
			</section>
		</div>
	);
}
