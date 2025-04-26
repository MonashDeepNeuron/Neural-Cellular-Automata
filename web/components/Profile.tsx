import Card from '@/components/Card';
import Image from 'next/image';



export interface ProfileData {
	name:string,
	imageLink:string,
	subtitle:string,
	description:string,
};

export default function ProfileCard({ name, imageLink, subtitle, description }: ProfileData) {
	return (
		<Card>
			<Image src={imageLink} alt={`${name}`} height={60} width={50} className='w-full rounded-full shadow aspect-square text-center' />
			<h2 className='text-2xl font-semibold text-purple-mdn mt-6'>{name}</h2>
			<h4 className='italic text-gray-600'>{subtitle}</h4>
			<p>{description}</p>
		</Card>
	);
}
