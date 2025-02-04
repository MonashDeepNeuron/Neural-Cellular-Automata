export default function CellularAutomata() {
	return (
		<div className='centre'>
			<h1>Cellular Automata Models</h1>

			<h2>Neural Cellular Automata</h2>
			<a href='/CAs/GCA/life.html' className='button' id='Growing Neural Cellular Automata'>
				G-NCA
			</a>
			<p>
				Neural Cellular Automata (NCA) are a category of cellular automata that involves using a neural network as the cell’s update rule.
				The neural network can be trained to figure out how to update the cell’s value in coordination with other cells operating on the
				same rule to produce a target behaviour.
			</p>
			<p>
				One of the best examples of NCA is{' '}
				<i>
					<a href='https://distill.pub/2020/growing-ca/'>Growing Neural Cellular Automata</a>
				</i>{' '}
				(A. Mordvintsev et. al, 2020) where they trained a Neural Cellular Automata to ‘grow’ into a target Images from a single seed cell.
			</p>
			<p>
				From a Deep Learning perspective, it can be characterised as a Recurrent Convolutional Neural Network. We describe what this means
				more in depth <a href='/Pages/nca-ca.html'>here</a>.
			</p>
			<p>
				Neural Cellular Automata display properties similar to how cells communicate and coordinate within living organisms. The model from
				Growing Neural Cellular Automata also naturally showed regenerative properties.
			</p>

			<h2>John Conway’s Game of Life / THE Game of Life</h2>
			<a href='/CAs/ConwaysLife/life.html' className='button' id='classic_conway'>
				Classic Conway
			</a>
			<p>This is probably the most famous example of cellular automata there is.</p>
			<p>The Game of Life operates on these very simple rules:</p>
			<ul>
				<li>All cells are either alive or dead (1 or 0).</li>
				<li>When a cell that is living has 2 or 3 neighbours, it gets to live.</li>
				<li>When a cell that is dead has 3 neighbours, it comes to life.</li>
				<li>In all other circumstances, the cell dies/remains dead.</li>
			</ul>
			<p>
				Even this simple rule can produce complex behaviours, and many patterns have been discovered that are self-sustaining. A more
				sophisticated version of the Game of Life can be found <a href='https://playgameoflife.com/'>here</a>.
			</p>

			<h2>Continuous Cellular Automata</h2>
			<a href='/CAs/Continuous/life.html' className='button' id='continuous'>
				Continuous
			</a>
			<p>
				Continuous CA also builds on Life Like CA. The main difference is that instead of using the binary dead or alive as states, we use a
				continuous range of values.
			</p>
		</div>
	);
}
