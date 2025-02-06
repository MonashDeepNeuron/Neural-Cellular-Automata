import clsx from 'clsx';
import type { HTMLProps } from 'react';

interface CardProps extends HTMLProps<HTMLDivElement> {
	children?: React.ReactNode;
}

export default function Card({ children, className, ...props }: CardProps) {
	return (
		<div className={clsx('bg-background rounded-md p-4 shadow-md', className)} {...props}>
			{children}
		</div>
	);
}
